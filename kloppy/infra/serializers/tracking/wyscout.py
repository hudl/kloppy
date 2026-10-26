from datetime import datetime, timedelta
import json
import logging
from typing import IO, NamedTuple, Optional, Union
import warnings

from kloppy.domain import (
    AttackingDirection,
    Ground,
    Metadata,
    Orientation,
    Period,
    Player,
    PlayerData,
    Point,
    Point3D,
    Provider,
    Team,
    TrackingDataset,
    attacking_directions_from_multi_frames,
)
from kloppy.domain.services.frame_factory import create_frame
from kloppy.infra.serializers.tracking.deserializer import (
    TrackingDataDeserializer,
)
from kloppy.utils import performance_logging

logger = logging.getLogger(__name__)

DEFAULT_FRAME_RATE = 25
DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0


class WyscoutTrackingInputs(NamedTuple):
    raw_data: IO[bytes]


class WyscoutTrackingDeserializer(
    TrackingDataDeserializer[WyscoutTrackingInputs]
):
    """
    Deserializer for Wyscout (broadcast) tracking data in JSONL format.

    The first line of the file is a header with the match metadata. Every
    following line is a single tracking frame. The file does not contain
    lineups, so teams and players are derived from the data: players are
    identified by their jersey number only.
    """

    def __init__(
        self,
        limit: Optional[int] = None,
        sample_rate: Optional[float] = None,
        coordinate_system: Optional[Union[str, Provider]] = None,
        pitch_length: float = DEFAULT_PITCH_LENGTH,
        pitch_width: float = DEFAULT_PITCH_WIDTH,
    ):
        super().__init__(limit, sample_rate, coordinate_system)
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

    @property
    def provider(self) -> Provider:
        return Provider.WYSCOUT

    @staticmethod
    def _create_teams(header: dict) -> list[Team]:
        """Create the home and away team.

        The header only contains a label such as "Arsenal - Chelsea, 2 - 1".
        If the team names can be read from it, the first one is assumed to be
        the home team. Otherwise, generic names are used.
        """
        names = {Ground.HOME: "Home", Ground.AWAY: "Away"}
        label = (header.get("metadata") or {}).get("label")
        if label:
            parts = label.rsplit(",", 1)[0].split(" - ")
            if len(parts) == 2 and all(p.strip() for p in parts):
                names = {
                    Ground.HOME: parts[0].strip(),
                    Ground.AWAY: parts[1].strip(),
                }

        return [
            Team(
                team_id=ground.value,
                name=names[ground],
                ground=ground,
            )
            for ground in (Ground.HOME, Ground.AWAY)
        ]

    @staticmethod
    def _get_player(
        team: Team, jersey_no: int, players: dict[tuple[str, int], Player]
    ) -> Player:
        """Get a player by jersey number, creating it on first sight."""
        key = (team.team_id, jersey_no)
        player = players.get(key)
        if player is None:
            player = Player(
                player_id=f"{team.team_id}_{jersey_no}",
                team=team,
                jersey_no=jersey_no,
            )
            team.players.append(player)
            players[key] = player
        return player

    @staticmethod
    def _parse_date(header: dict) -> Optional[datetime]:
        scheduled_time = (header.get("metadata") or {}).get("scheduledTime")
        if not scheduled_time:
            return None
        try:
            return datetime.fromisoformat(scheduled_time.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _parse_frame(
        self,
        raw_frame: dict,
        period: Period,
        teams: list[Team],
        players: dict[tuple[str, int], Player],
    ):
        ball_coordinates = None
        ball_speed = None
        other_data = {}
        ball = next((b for b in raw_frame["balls"] if b), None)
        if ball is not None:
            ball_coordinates = Point3D(
                x=ball["x"], y=ball["y"], z=ball.get("z") or 0.0
            )
            ball_speed = ball.get("speed")
            other_data = {
                "ball_vx": ball.get("vx"),
                "ball_vy": ball.get("vy"),
                "ball_visibility": ball.get("visibility"),
                "ball_confidence": ball.get("confidence"),
            }

        players_data = {}
        for team, key in zip(teams, ("homePlayers", "awayPlayers")):
            for raw_player in raw_frame[key]:
                if not raw_player:
                    continue
                player = self._get_player(
                    team, int(raw_player["jerseyNum"]), players
                )
                players_data[player] = PlayerData(
                    coordinates=Point(x=raw_player["x"], y=raw_player["y"]),
                    speed=raw_player.get("speed"),
                    other_data={
                        "vx": raw_player.get("vx"),
                        "vy": raw_player.get("vy"),
                        "visibility": raw_player.get("visibility"),
                        "confidence": raw_player.get("confidence"),
                    },
                )

        return create_frame(
            frame_id=raw_frame["frameNum"],
            timestamp=timedelta(seconds=raw_frame["periodElapsedTime"]),
            ball_coordinates=ball_coordinates,
            ball_speed=ball_speed,
            players_data=players_data,
            period=period,
            ball_state=None,
            ball_owning_team=None,
            other_data=other_data,
        )

    def deserialize(self, inputs: WyscoutTrackingInputs) -> TrackingDataset:
        transformer = self.get_transformer(
            pitch_length=self.pitch_length, pitch_width=self.pitch_width
        )

        header = None
        teams = None
        players: dict[tuple[str, int], Player] = {}
        periods: dict[int, Period] = {}
        frames = []
        video_times = []
        n = 0
        sample = 1.0 / self.sample_rate

        with performance_logging("Loading data", logger=logger):
            for line in inputs.raw_data:
                if not line.strip():
                    continue
                raw_frame = json.loads(line)

                if "frameNum" not in raw_frame:
                    header = raw_frame
                    continue
                if teams is None:
                    teams = self._create_teams(header or {})

                period_id = raw_frame["period"]
                video_time = raw_frame["videoTimeMs"]
                elapsed = raw_frame["periodElapsedTime"]

                period = periods.get(period_id)
                if period is None:
                    if elapsed <= 0:
                        # The period clock has not started yet, so this
                        # frame does not belong to any period.
                        continue
                    period = Period(
                        id=period_id,
                        start_timestamp=timedelta(
                            milliseconds=video_time - elapsed * 1000
                        ),
                        end_timestamp=timedelta(milliseconds=video_time),
                    )
                    periods[period_id] = period
                else:
                    period.end_timestamp = timedelta(milliseconds=video_time)

                if len(video_times) < 2:
                    video_times.append(video_time)

                if n % sample == 0:
                    frame = self._parse_frame(raw_frame, period, teams, players)
                    frames.append(transformer.transform_frame(frame))

                n += 1
                if self.limit and n >= self.limit:
                    break

        if header is None or teams is None or not frames:
            raise ValueError("No tracking data found in the input file")

        frame_rate = DEFAULT_FRAME_RATE
        if len(video_times) == 2 and video_times[1] > video_times[0]:
            frame_rate = round(1000 / (video_times[1] - video_times[0]))

        periods_list = list(periods.values())
        attacking_directions = attacking_directions_from_multi_frames(
            frames, periods_list
        )
        first_direction = attacking_directions[periods_list[0].id]
        if first_direction == AttackingDirection.LTR:
            orientation = Orientation.HOME_AWAY
        elif first_direction == AttackingDirection.RTL:
            orientation = Orientation.AWAY_HOME
        else:
            warnings.warn(
                "Could not determine orientation of dataset, defaulting to NOT_SET"
            )
            orientation = Orientation.NOT_SET

        coordinate_system = transformer.get_to_coordinate_system()
        metadata = Metadata(
            teams=teams,
            periods=periods_list,
            frame_rate=frame_rate,
            orientation=orientation,
            pitch_dimensions=coordinate_system.pitch_dimensions,
            coordinate_system=coordinate_system,
            flags=None,
            provider=Provider.WYSCOUT,
            game_id=header.get("gameRefId"),
            date=self._parse_date(header),
        )

        return TrackingDataset(records=frames, metadata=metadata)
