# behavior_manager.py
class BehaviorState:
    GO = "DRIVE"
    SLOW = "SLOWING"
    STOP = "STOPPED"

class BehaviorManager:
    def __init__(self):
        self.state = BehaviorState.GO
        self.history = {}
        self.dt = 0.1

    def update_state(self, obstacles, lidar_dist):
        ttc = 99.0
        
        for obs in obstacles:
            oid = obs['id']
            curr_d = obs['dist']
            if oid in self.history:
                # Calculate how fast the object is closing in
                rel_v = (self.history[oid] - curr_d) / self.dt
                if rel_v > 0:
                    ttc = min(ttc, curr_d / rel_v)
            self.history[oid] = curr_d

        # Simple Decision Logic
        if lidar_dist < 2.0 or ttc < 1.0:
            self.state = BehaviorState.STOP
            return f"BREAK! 100% (TTC: {ttc:.1f}s)"

        elif lidar_dist < 6.0 or ttc < 3.0:
            self.state = BehaviorState.SLOW
            return f"SLOWING (TTC: {ttc:.1f}s)"

        else:
            self.state = BehaviorState.GO
            return "DRIVING"