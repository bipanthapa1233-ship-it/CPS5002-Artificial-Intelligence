from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt 
Position = Tuple[int, int]


# === ENUMS & BASIC TYPES ======================================================

class Role(Enum):
    DEK = "Dek"
    FATHER = "Father"
    BROTHER = "Brother"
    CLAN = "Clan"
    WILDLIFE = "Wildlife"
    ADVERSARY = "Adversary"
    SYNTHETIC = "Synthetic"


class CellTerrain(Enum):
    EMPTY = auto()
    ROCKY = auto()
    CANYON = auto()
    HAZARD = auto()
    TRAP = auto()
    TELEPORT = auto()


# === STATS & ENTITY CLASSES ===================================================

@dataclass
class Stats:
    max_health: int
    health: int
    max_stamina: int
    stamina: int
    honour: int = 0

    def is_alive(self) -> bool:
        return self.health > 0

    def apply_damage(self, amount: int) -> None:
        self.health = max(0, self.health - amount)

    def recover_stamina(self, amount: int) -> None:
        self.stamina = min(self.max_stamina, self.stamina + amount)

    def spend_stamina(self, amount: int) -> bool:
        """Spend stamina if available, otherwise fail."""
        if self.stamina >= amount:
            self.stamina -= amount
            return True
        return False


@dataclass
class Entity:
    id: int
    name: str
    role: Role
    position: Position
    stats: Stats
    carrying: Optional["Entity"] = None
    is_hostile_to_dek: bool = False

    def is_alive(self) -> bool:
        return self.stats.is_alive()

    def move_to(self, new_pos: Position) -> None:
        self.position = new_pos

    def __repr__(self) -> str:
        return (
            f"<{self.role.value} {self.name} at {self.position} "
            f"HP:{self.stats.health} ST:{self.stats.stamina} H:{self.stats.honour}>"
        )


@dataclass
class GridCell:
    terrain: CellTerrain = CellTerrain.EMPTY
    entities: List[Entity] = field(default_factory=list)

    def add_entity(self, e: Entity) -> None:
        if e not in self.entities:
            self.entities.append(e)

    def remove_entity(self, e: Entity) -> None:
        if e in self.entities:
            self.entities.remove(e)

    def has_hazard(self) -> bool:
        return self.terrain in {CellTerrain.HAZARD, CellTerrain.TRAP}


# === GRID WORLD ===============================================================

class GridWorld:
    """
    2-D toroidal grid world for the planet Kalisk.
    """

    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.grid: List[List[GridCell]] = [
            [GridCell() for _ in range(width)] for _ in range(height)
        ]
        self.entities: Dict[int, Entity] = {}
        self.next_id = 1

        self._procedurally_generate_terrain()

    def _procedurally_generate_terrain(self) -> None:
        """
        Simple procedural generation of terrain:
        - Mostly EMPTY
        - Some ROCKY/CANYON
        - Sparse HAZARD/TRAP/TELEPORT for survival constraints
        """
        for y in range(self.height):
            for x in range(self.width):
                r = random.random()
                if r < 0.65:
                    terrain = CellTerrain.EMPTY
                elif r < 0.8:
                    terrain = CellTerrain.ROCKY
                elif r < 0.9:
                    terrain = CellTerrain.CANYON
                elif r < 0.96:
                    terrain = CellTerrain.HAZARD
                elif r < 0.99:
                    terrain = CellTerrain.TRAP
                else:
                    terrain = CellTerrain.TELEPORT
                self.grid[y][x].terrain = terrain

    def wrap_pos(self, pos: Position) -> Position:
        x, y = pos
        return x % self.width, y % self.height

    def get_cell(self, pos: Position) -> GridCell:
        x, y = self.wrap_pos(pos)
        return self.grid[y][x]

    def neighbours(self, pos: Position) -> List[Position]:
        x, y = pos
        candidates = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
        return [self.wrap_pos(p) for p in candidates]

    def spawn_entity(
        self,
        name: str,
        role: Role,
        position: Optional[Position] = None,
        max_health: int = 100,
        max_stamina: int = 100,
        honour: int = 0,
        hostile: bool = False,
    ) -> Entity:
        if position is None:
            position = (random.randrange(self.width), random.randrange(self.height))
        stats = Stats(
            max_health=max_health,
            health=max_health,
            max_stamina=max_stamina,
            stamina=max_stamina,
            honour=honour,
        )
        e = Entity(
            id=self.next_id,
            name=name,
            role=role,
            position=self.wrap_pos(position),
            stats=stats,
            carrying=None,
            is_hostile_to_dek=hostile,
        )
        self.next_id += 1
        self.entities[e.id] = e
        self.get_cell(e.position).add_entity(e)
        return e

    def move_entity(self, e: Entity, new_pos: Position) -> None:
        cell_old = self.get_cell(e.position)
        cell_old.remove_entity(e)
        e.move_to(self.wrap_pos(new_pos))
        cell_new = self.get_cell(e.position)
        cell_new.add_entity(e)

    def remove_dead(self) -> None:
        dead_ids = [eid for eid, e in self.entities.items() if not e.is_alive()]
        for eid in dead_ids:
            e = self.entities[eid]
            self.get_cell(e.position).remove_entity(e)
            del self.entities[eid]

    def random_empty_position(self) -> Position:
        while True:
            pos = (random.randrange(self.width), random.randrange(self.height))
            if not self.get_cell(pos).entities:
                return pos

    def find_entities_by_role(self, role: Role) -> List[Entity]:
        return [e for e in self.entities.values() if e.role == role]

    def print_ascii(self, dek: Optional[Entity] = None) -> None:
        """
        Simple ASCII visualisation for debugging and screenshots.
        D = Dek
        A = Adversary
        m = some other entity
        terrain symbols for empty cells
        """
        symbol_map = {
            CellTerrain.EMPTY: ".",
            CellTerrain.ROCKY: "^",
            CellTerrain.CANYON: "v",
            CellTerrain.HAZARD: "!",
            CellTerrain.TRAP: "x",
            CellTerrain.TELEPORT: "O",
        }
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                cell = self.grid[y][x]
                if cell.entities:
                    # Prioritise showing Dek or Adversary
                    dek_here = [e for e in cell.entities if e.role == Role.DEK]
                    adv_here = [e for e in cell.entities if e.role == Role.ADVERSARY]
                    if dek_here:
                        row += "D"
                    elif adv_here:
                        row += "A"
                    else:
                        row += "m"
                else:
                    row += symbol_map[cell.terrain]
            print(row)
        if dek is not None:
            print(
                f"Dek: HP={dek.stats.health} "
                f"ST={dek.stats.stamina} Honour={dek.stats.honour}"
            )


# === CLAN CODE / HONOUR SYSTEM ===============================================

class ClanCode:
    """Implements the Yautja honour rules in a simplified way."""

    @staticmethod
    def is_worthy_target(target: Entity) -> bool:
        if target.role == Role.WILDLIFE:
            return True
        if target.role == Role.ADVERSARY:
            return True
        if target.role == Role.CLAN:
            # Could be honour trial
            return True
        return False

    @staticmethod
    def honour_change_for_kill(target: Entity) -> int:
        if target.role == Role.WILDLIFE:
            return 5
        if target.role == Role.CLAN:
            return 10
        if target.role == Role.ADVERSARY:
            return 100
        return 0

    @staticmethod
    def dishonour_for_unworthy(target: Entity) -> int:
        # For simplicity, anything else is unworthy (e.g., helpless synthetics).
        return -50


# === DEK POLICY (ON-LINE LEARNING) ===========================================

class DekPolicy:
    """
    Very simple on-line learning policy for Dek (reinforcement-like, no external libs).

    State is binned by:
      - distance to adversary (near/far)
      - stamina level (low/high)
      - whether Dek is with Thia or alone

    Actions:
      - move_to_boss
      - hunt
      - rest
      - seek_resources
    """

    ACTIONS = ["move_to_boss", "hunt", "rest", "seek_resources"]

    def __init__(self, alpha: float = 0.3, gamma: float = 0.9, epsilon: float = 0.1):
        self.q_table: Dict[Tuple[str, str], float] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.last_state: Optional[str] = None
        self.last_action: Optional[str] = None

    def _encode_state(
        self,
        dek: Entity,
        adversary: Entity,
        thia: Optional[Entity],
        grid: GridWorld,
    ) -> str:
        # Toroidal Manhattan distance
        dx = min(
            abs(dek.position[0] - adversary.position[0]),
            grid.width - abs(dek.position[0] - adversary.position[0]),
        )
        dy = min(
            abs(dek.position[1] - adversary.position[1]),
            grid.height - abs(dek.position[1] - adversary.position[1]),
        )
        distance = dx + dy
        dist_bucket = "near" if distance <= 5 else "far"
        stamina_bucket = (
            "low" if dek.stats.stamina < dek.stats.max_stamina * 0.3 else "high"
        )
        thia_with = "with" if thia and thia.position == dek.position else "alone"
        return f"{dist_bucket}|{stamina_bucket}|{thia_with}"

    def select_action(
        self,
        dek: Entity,
        adversary: Entity,
        thia: Optional[Entity],
        grid: GridWorld,
    ) -> str:
        state = self._encode_state(dek, adversary, thia, grid)
        # epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(self.ACTIONS)
        else:
            qs = {a: self.q_table.get((state, a), 0.0) for a in self.ACTIONS}
            max_q = max(qs.values())
            best_actions = [a for a, q in qs.items() if q == max_q]
            action = random.choice(best_actions)
        self.last_state = state
        self.last_action = action
        return action

    def update(
        self,
        reward: float,
        dek: Entity,
        adversary: Entity,
        thia: Optional[Entity],
        grid: GridWorld,
    ) -> None:
        if self.last_state is None or self.last_action is None:
            return
        prev_state = self.last_state
        prev_action = self.last_action
        new_state = self._encode_state(dek, adversary, thia, grid)
        # Q-learning update (tabular)
        prev_q = self.q_table.get((prev_state, prev_action), 0.0)
        future_qs = [self.q_table.get((new_state, a), 0.0) for a in self.ACTIONS]
        best_future_q = max(future_qs) if future_qs else 0.0
        updated_q = prev_q + self.alpha * (reward + self.gamma * best_future_q - prev_q)
        self.q_table[(prev_state, prev_action)] = updated_q


# === ADAPTIVE ADVERSARY BRAIN ================================================

class AdversaryBrain:
    """
    Simple adaptive behaviour for the ultimate adversary.

    Tracks patterns in Dek's chosen high-level actions and becomes
    more aggressive at countering repeated strategies.
    """

    def __init__(self):
        self.dek_action_counts: Dict[str, int] = {a: 0 for a in DekPolicy.ACTIONS}
        self.aggression_level: float = 1.0

    def observe_dek_action(self, action: str) -> None:
        if action in self.dek_action_counts:
            self.dek_action_counts[action] += 1
        total = sum(self.dek_action_counts.values())
        if total > 0:
            dominant = max(self.dek_action_counts.values())
            ratio = dominant / total
            # As one tactic dominates, aggression scales from 1.0 -> 2.0
            self.aggression_level = 1.0 + ratio

    def choose_action(self, adversary: Entity, dek: Entity, grid: GridWorld) -> str:
        """
        Choose between chasing Dek or patrolling, influenced by aggression level.
        """
        if random.random() < self.aggression_level / 2.0:
            return "chase_dek"
        return "patrol"


# === SIMULATION ENGINE =======================================================

class Simulation:
    """
    Main simulation engine tying environment, agents, and dynamics together.
    """

    def __init__(self, width: int = 20, height: int = 20, max_steps: int = 500):
        self.grid = GridWorld(width, height)
        self.max_steps = max_steps
        self.current_step = 0

        # -- Core actors ------------------------------------------------------
        self.dek: Entity = self.grid.spawn_entity(
            "Dek",
            Role.DEK,
            position=self.grid.random_empty_position(),
            max_health=120,
            max_stamina=120,
            honour=0,
        )
        self.father: Entity = self.grid.spawn_entity(
            "Father",
            Role.FATHER,
            position=self.grid.random_empty_position(),
            max_health=150,
            max_stamina=110,
            honour=100,
            hostile=True,
        )
        self.brother: Entity = self.grid.spawn_entity(
            "Brother",
            Role.BROTHER,
            position=self.grid.random_empty_position(),
            max_health=130,
            max_stamina=110,
            honour=50,
            hostile=True,
        )
        # Clan members (social mechanics)
        for i in range(3):
            self.grid.spawn_entity(
                f"Clan_{i+1}",
                Role.CLAN,
                position=self.grid.random_empty_position(),
                max_health=100,
                max_stamina=100,
                honour=30,
                hostile=False,
            )

        # Thia (damaged synthetic ally)
        self.thia: Entity = self.grid.spawn_entity(
            "Thia",
            Role.SYNTHETIC,
            position=self.grid.random_empty_position(),
            max_health=80,
            max_stamina=0,
            honour=0,
        )
        self.thia_damaged: bool = True

        # Wildlife / monsters
        for i in range(15):
            self.grid.spawn_entity(
                f"Beast_{i+1}",
                Role.WILDLIFE,
                position=self.grid.random_empty_position(),
                max_health=40,
                max_stamina=60,
                hostile=True,
            )

        # Ultimate adversary (boss)
        self.adversary: Entity = self.grid.spawn_entity(
            "Kaiju",
            Role.ADVERSARY,
            position=self.grid.random_empty_position(),
            max_health=400,
            max_stamina=150,
            hostile=True,
        )

        # Brains
        self.dek_policy = DekPolicy()
        self.adversary_brain = AdversaryBrain()

        # Logging for report (for graphs, stats)
        self.history: List[Dict[str, float]] = []

    # --- Utility methods -----------------------------------------------------

    def manhattan_distance(self, a: Position, b: Position) -> int:
        """Toroidal Manhattan distance."""
        dx = min(abs(a[0] - b[0]), self.grid.width - abs(a[0] - b[0]))
        dy = min(abs(a[1] - b[1]), self.grid.height - abs(a[1] - b[1]))
        return dx + dy

    def find_closest(self, source: Entity, predicate) -> Optional[Entity]:
        candidates = [
            e
            for e in self.grid.entities.values()
            if e.is_alive() and e is not source and predicate(e)
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda e: self.manhattan_distance(source.position, e.position),
        )

    # --- One simulation step -------------------------------------------------

    def step(self) -> None:
        if not self.dek.is_alive():
            return
        if not self.adversary.is_alive():
            return

        self.current_step += 1

        # Dek + Thia cooperative high-level decision
        action = self.dek_policy.select_action(
            self.dek, self.adversary, self.thia, self.grid
        )
        self.adversary_brain.observe_dek_action(action)

        honour_before = self.dek.stats.honour
        health_before = self.dek.stats.health

        # Execute Dek's high-level action
        if action == "move_to_boss":
            self._dek_move_towards(self.adversary.position)
        elif action == "hunt":
            self._dek_hunt()
        elif action == "rest":
            self._dek_rest()
        elif action == "seek_resources":
            self._dek_seek_resources()

        # Thia behaviour (support / knowledge)
        self._thia_behaviour()

        # Clan behaviour (social mechanics, honour duels)
        self._clan_behaviour()

        # Wildlife behaviour (threats / survival)
        self._wildlife_behaviour()

        # Adversary behaviour (adaptive, uses AdversaryBrain)
        self._adversary_behaviour()

        # Environmental hazards (traps, hazardous cells, teleport)
        self._apply_environment_hazards()

        # Remove dead entities from grid
        self.grid.remove_dead()

        # Reward for Dek's learning
        honour_delta = self.dek.stats.honour - honour_before
        health_delta = self.dek.stats.health - health_before
        reward = honour_delta + (health_delta * 0.1)
        if not self.dek.is_alive():
            reward -= 100
        if not self.adversary.is_alive():
            reward += 200

        self.dek_policy.update(reward, self.dek, self.adversary, self.thia, self.grid)

        # Log metrics
        self.history.append(
            {
                "step": float(self.current_step),
                "dek_health": float(self.dek.stats.health),
                "dek_stamina": float(self.dek.stats.stamina),
                "dek_honour": float(self.dek.stats.honour),
                "adversary_health": float(self.adversary.stats.health),
            }
        )

    # --- Movement helper -----------------------------------------------------

    def _move_entity_towards(self, e: Entity, target_pos: Position) -> None:
        if not e.stats.spend_stamina(2):
            return
        x, y = e.position
        tx, ty = target_pos
        # choose direction that minimises wrapped distance
        choices = []
        # horizontal
        if x != tx:
            step = (
                1
                if ((tx - x) % self.grid.width) < (self.grid.width // 2)
                else -1
            )
            choices.append(
                ((x + step, y), self.manhattan_distance((x + step, y), target_pos))
            )
        # vertical
        if y != ty:
            step = (
                1
                if ((ty - y) % self.grid.height) < (self.grid.height // 2)
                else -1
            )
            choices.append(
                ((x, y + step), self.manhattan_distance((x, y + step), target_pos))
            )
        if not choices:
            return
        new_pos, _ = min(choices, key=lambda c: c[1])
        self.grid.move_entity(e, new_pos)

    # --- Dek behaviours ------------------------------------------------------

    def _dek_move_towards(self, target_pos: Position) -> None:
        # If Thia is damaged and close, Dek can carry her (affects stamina)
        if self.thia_damaged and self.manhattan_distance(
            self.dek.position, self.thia.position
        ) <= 1:
            self.dek.carrying = self.thia
            self.thia.position = self.dek.position

        self._move_entity_towards(self.dek, target_pos)

        # Carrying Thia costs extra stamina and keeps her with Dek
        if self.dek.carrying is not None:
            self.dek.stats.spend_stamina(1)
            self.thia.position = self.dek.position

    def _dek_hunt(self) -> None:
        # Hunt wildlife or participate in clan trials
        target = self.find_closest(
            self.dek, lambda e: e.role in {Role.WILDLIFE, Role.CLAN}
        )
        if not target:
            return
        if self.manhattan_distance(self.dek.position, target.position) > 1:
            self._move_entity_towards(self.dek, target.position)
        else:
            # In melee range
            self._melee_attack(self.dek, target)

    def _dek_rest(self) -> None:
        # Rest to regain stamina and a bit of health
        self.dek.stats.recover_stamina(10)
        if self.dek.stats.health < self.dek.stats.max_health:
            self.dek.stats.health += 1

    def _dek_seek_resources(self) -> None:
        # heuristic: move towards a random safe-ish cell (EMPTY/ROCKY/TELEPORT)
        safe_positions = []
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell = self.grid.grid[y][x]
                if cell.terrain in {
                    CellTerrain.EMPTY,
                    CellTerrain.ROCKY,
                    CellTerrain.TELEPORT,
                }:
                    safe_positions.append((x, y))
        if not safe_positions:
            return
        target_pos = random.choice(safe_positions)
        self._move_entity_towards(self.dek, target_pos)

    # --- Support behaviours (Thia, Clan, Wildlife, Adversary) ----------------

    def _thia_behaviour(self) -> None:
        if not self.thia.is_alive():
            return
        # If carried, just stay with Dek
        if self.dek.carrying is self.thia:
            self.thia.position = self.dek.position
            return

        # Recon / knowledge: if adversary is fairly near, she gives a small honour nudge
        distance = self.manhattan_distance(self.dek.position, self.adversary.position)
        if 2 < distance < 8:
            # Simulate Thia warning Dek (small honour bonus to encourage survivable distance)
            self.dek.stats.honour += 1

        # If capable, Thia slowly crawls toward Dek to maintain alliance
        if self.manhattan_distance(self.dek.position, self.thia.position) > 2:
            self._move_entity_towards(self.thia, self.dek.position)

    def _clan_behaviour(self) -> None:
        clan_members = [
            e
            for e in self.grid.entities.values()
            if e.role in {Role.FATHER, Role.BROTHER, Role.CLAN}
        ]
        for member in clan_members:
            if not member.is_alive():
                continue
            dist = self.manhattan_distance(member.position, self.dek.position)
            if dist <= 1 and member.role in {Role.FATHER, Role.BROTHER}:
                # Honour duel: tests Dek's worth
                self._melee_attack(member, self.dek, honour_duel=True)
            else:
                # Patrol or wander (social presence)
                if random.random() < 0.5:
                    neighbours = self.grid.neighbours(member.position)
                    new_pos = random.choice(neighbours)
                    if member.stats.spend_stamina(1):
                        self.grid.move_entity(member, new_pos)

    def _wildlife_behaviour(self) -> None:
        wildlife = [
            e for e in self.grid.entities.values() if e.role == Role.WILDLIFE
        ]
        for beast in wildlife:
            if not beast.is_alive():
                continue
            dist_dek = self.manhattan_distance(beast.position, self.dek.position)
            if dist_dek <= 1:
                self._melee_attack(beast, self.dek)
            else:
                # Wander randomly (environmental threat)
                if random.random() < 0.7:
                    neighbours = self.grid.neighbours(beast.position)
                    new_pos = random.choice(neighbours)
                    if beast.stats.spend_stamina(1):
                        self.grid.move_entity(beast, new_pos)

    def _adversary_behaviour(self) -> None:
        if not self.adversary.is_alive():
            return
        action = self.adversary_brain.choose_action(
            self.adversary, self.dek, self.grid
        )
        if action == "chase_dek":
            self._move_entity_towards(self.adversary, self.dek.position)
            if self.manhattan_distance(
                self.adversary.position, self.dek.position
            ) <= 1:
                self._melee_attack(self.adversary, self.dek)
        elif action == "patrol":
            # Patrol slowly
            if random.random() < 0.5:
                neighbours = self.grid.neighbours(self.adversary.position)
                new_pos = random.choice(neighbours)
                if self.adversary.stats.spend_stamina(1):
                    self.grid.move_entity(self.adversary, new_pos)

    # --- Environmental hazards / survival ------------------------------------

    def _apply_environment_hazards(self) -> None:
        for e in list(self.grid.entities.values()):
            cell = self.grid.get_cell(e.position)
            if cell.terrain == CellTerrain.HAZARD:
                e.stats.apply_damage(2)
                e.stats.spend_stamina(1)
            elif cell.terrain == CellTerrain.TRAP:
                # One-time heavy damage, then disable trap
                e.stats.apply_damage(15)
                e.stats.spend_stamina(5)
                cell.terrain = CellTerrain.EMPTY
            elif cell.terrain == CellTerrain.TELEPORT:
                # Chance to teleport to a random safe cell
                if random.random() < 0.2:
                    new_pos = self.grid.random_empty_position()
                    self.grid.move_entity(e, new_pos)

    # --- Combat / honour mechanics -------------------------------------------

    def _melee_attack(
        self,
        attacker: Entity,
        defender: Entity,
        honour_duel: bool = False,
    ) -> None:
        if not attacker.stats.spend_stamina(5):
            return
        # Base damage by role
        base_damage = 10
        if attacker.role == Role.ADVERSARY:
            base_damage = 25
        elif attacker.role == Role.DEK:
            base_damage = 15
        damage = base_damage + random.randint(-3, 3)
        defender.stats.apply_damage(damage)

        # Honour system for Dek
        if attacker.role == Role.DEK:
            if ClanCode.is_worthy_target(defender):
                if not defender.is_alive():
                    attacker.stats.honour += ClanCode.honour_change_for_kill(defender)
                else:
                    attacker.stats.honour += 1
            else:
                attacker.stats.honour += ClanCode.dishonour_for_unworthy(defender)

        if honour_duel:
            # Duel affects honour if Dek defeats father/brother or vice versa
            if not defender.is_alive() and attacker.role in {Role.FATHER, Role.BROTHER}:
                # Dek is defender here: he has lost
                if defender.role == Role.DEK:
                    attacker.stats.honour += 40
                else:
                    self.dek.stats.honour += 40

    # --- Run the simulation ---------------------------------------------------

    def run(self, verbose: bool = False) -> Dict[str, float]:
        while (
            self.current_step < self.max_steps
            and self.dek.is_alive()
            and self.adversary.is_alive()
        ):
            if verbose and self.current_step % 20 == 0:
                print(f"=== Step {self.current_step} ===")
                self.grid.print_ascii(self.dek)
            self.step()

        # --- NEW: nice textual summary for report / screenshots ---------------
        outcome_type = "Ongoing"
        if self.dek.is_alive() and not self.adversary.is_alive():
            outcome_type = "Dek defeated the ultimate adversary"
        elif not self.dek.is_alive() and self.adversary.is_alive():
            outcome_type = "Dek died before defeating the adversary"
        elif not self.dek.is_alive() and not self.adversary.is_alive():
            outcome_type = "Mutual destruction"

        if verbose:
            print("\nSimulation finished.")
            self.grid.print_ascii(self.dek)
            print("\n--- Run Summary ---")
            print(f"Total steps: {self.current_step}")
            print(f"Outcome: {outcome_type}")
            print(f"Dek alive: {self.dek.is_alive()} (HP={self.dek.stats.health})")
            print(
                f"Adversary alive: {self.adversary.is_alive()} "
                f"(HP={self.adversary.stats.health})"
            )
            print(f"Final Dek honour: {self.dek.stats.honour}")
            print(
                "Interpretation: Dek's honour reflects successful hunts, "
                "clan trials, and whether he reached the final adversary."
            )

        outcome = {
            "steps": float(self.current_step),
            "dek_alive": float(1 if self.dek.is_alive() else 0),
            "adversary_alive": float(1 if self.adversary.is_alive() else 0),
            "dek_honour": float(self.dek.stats.honour),
        }
        return outcome

    # --- NEW: plotting for report --------------------------------------------

    def plot_metrics(self, prefix: str = "run1") -> None:
        """
        Create simple PNG plots for:
        - Dek's honour vs steps
        - Adversary health vs steps

        These figures can go directly into the report.
        """
        if not self.history:
            print("No history to plot.")
            return

        steps = [h["step"] for h in self.history]
        honour = [h["dek_honour"] for h in self.history]
        adv_health = [h["adversary_health"] for h in self.history]

        # Honour vs steps
        plt.figure()
        plt.plot(steps, honour)
        plt.xlabel("Simulation Step")
        plt.ylabel("Dek Honour")
        plt.title("Dek Honour Progression Over Time")
        plt.tight_layout()
        plt.savefig(f"{prefix}_honour_vs_steps.png")
        plt.close()

        # Adversary health vs steps
        plt.figure()
        plt.plot(steps, adv_health)
        plt.xlabel("Simulation Step")
        plt.ylabel("Adversary Health")
        plt.title("Adversary Health Over Time")
        plt.tight_layout()
        plt.savefig(f"{prefix}_adversary_health_vs_steps.png")
        plt.close()

        print(
            f"Saved plots:\n"
            f"  {prefix}_honour_vs_steps.png\n"
            f"  {prefix}_adversary_health_vs_steps.png"
        )


# === MULTI-RUN EXPERIMENTS (FOR REPORT) ======================================

def run_multiple_simulations(runs: int = 20) -> None:
    """
    Runs the simulation multiple times and prints simple aggregate statistics.
    Use this output in your report for graphs / quantitative analysis.
    """
    results = []
    for i in range(runs):
        sim = Simulation()
        outcome = sim.run(verbose=False)
        results.append(outcome)

    avg_steps = sum(r["steps"] for r in results) / len(results)
    wins = [r for r in results if r["adversary_alive"] == 0.0]
    losses = [r for r in results if r["dek_alive"] == 0.0 and r["adversary_alive"] == 1.0]

    win_rate = len(wins) / len(results)
    loss_rate = len(losses) / len(results)
    avg_honour = sum(r["dek_honour"] for r in results) / len(results)

    print("\n=== Batch Simulation Statistics ===")
    print(f"Number of runs: {len(results)}")
    print(f"Average steps per run: {avg_steps:.1f}")
    print(f"Dek victory rate (adversary defeated): {win_rate * 100:.1f}%")
    print(f"Dek death rate (adversary survives): {loss_rate * 100:.1f}%")
    print(f"Average final Dek honour: {avg_honour:.1f}")
    print(
        "Interpretation: A low victory rate with moderate honour suggests "
        "Kalisk is extremely hostile and Dek often dies before reaching the boss."
    )


# === MAIN ENTRYPOINT =========================================================

if __name__ == "__main__":
    sim = Simulation()
    sim.run(verbose=True)
    sim.plot_metrics(prefix="run1")
    run_multiple_simulations(10)
