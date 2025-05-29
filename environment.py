import random, torch
from core import device, OFFS, DIRS, batched_update_fire_exit

class Agent:
    def __init__(self, idx, grid, size, speed, occupied):
        self.id=idx; self.size=size; self.speed=speed; self.grid=grid
        self.position=self._random_pos(occupied); self.health=100
        self.alive=True; self.exited=False; self.knows_exit=(idx==0)
    def _random_pos(self, occ):
        while True:
            x=random.randint(0,self.grid-self.size)
            y=random.randint(0,self.grid-self.size)
            cells={(x+i,y+j) for i in range(self.size) for j in range(self.size)}
            if not (cells&occ): return (x,y)
    def cells(self):
        x,y=self.position
        return {(x+i,y+j) for i in range(self.size) for j in range(self.size)}

class Environment:
    def __init__(self, grid, agent_specs, num_fires, exit_pos, num_leaders=0):
        self.grid=grid; self.exit_positions=exit_pos; self.num_fires=num_fires
        self.exit_mask=torch.zeros((grid,grid),dtype=torch.bool,device=device)
        for x,y in exit_pos: self.exit_mask[y,x]=True

        self.num_leaders = num_leaders
        self.leader_size = 3

        # агенты
        self.N=sum(c for c,_ in agent_specs)
        self.size=torch.empty(self.N,device=device)
        self.speed=torch.empty(self.N,device=device)
        i=0
        for cnt,sz in agent_specs:
            for _ in range(cnt):
                self.size[i]=sz/5.; self.speed[i]=float(sz); i+=1
        self.knows_exit=torch.zeros(self.N,dtype=torch.bool,device=device); self.knows_exit[0]=True
        # runtime-тензоры
        self.positions=torch.zeros((self.N,2),dtype=torch.long,device=device)
        self.alive=torch.ones(self.N,dtype=torch.bool,device=device)
        self.health=torch.full((self.N,),100.,device=device)
        self.fire_mask=torch.zeros((grid,grid),dtype=torch.float32,device=device)

        # лидеры
        self.leader_positions = torch.zeros((self.num_leaders, 2), dtype=torch.long, device=device)
        self.leader_alive     = torch.ones((self.num_leaders,), dtype=torch.bool, device=device)

        self.reset()

    def reset(self):
        self.alive.fill_(True); self.health.fill_(100.)
        self.positions[:,0]=torch.randint(0,self.grid,(self.N,),device=device)
        self.positions[:,1]=torch.randint(0,self.grid,(self.N,),device=device)
        maxc=self.grid-(self.size*5).round().long()
        self.positions[:,0]=torch.min(self.positions[:,0],maxc)
        self.positions[:,1]=torch.min(self.positions[:,1],maxc)

        self.fire_mask.zero_()
        SAFE_RADIUS = self.grid // 2
        safe_x = self.grid - SAFE_RADIUS
        safe_y = self.grid - SAFE_RADIUS

        while int(self.fire_mask.sum()) < self.num_fires:
            x=random.randint(0,self.grid-1)
            y=random.randint(0,self.grid-1)
            if (x,y) not in self.exit_positions and (x < safe_x or y < safe_y):
                self.fire_mask[y,x]=1.

        # лидеры — случайные позиции
        if self.num_leaders > 0:
            max_leader = self.grid - self.leader_size
            self.leader_positions[:, 0] = torch.randint(0, max_leader + 1, (self.num_leaders,), device=device)
            self.leader_positions[:, 1] = torch.randint(0, max_leader + 1, (self.num_leaders,), device=device)
            self.leader_alive.fill_(True)

    def step_leaders(self):
        """Прямолинейное движение лидеров к ближайшему выходу"""
        if self.num_leaders == 0:
            return

        for i in range(self.num_leaders):
            if not self.leader_alive[i]:
                continue
            pos = self.leader_positions[i].float()
            exit_coords = torch.tensor(self.exit_positions, dtype=torch.float32, device=device)
            dists = torch.norm(exit_coords - pos, dim=1)
            target = exit_coords[dists.argmin()]
            direction = (target - pos).sign().to(torch.long)  # [dx, dy] ∈ {-1, 0, 1}
            new_pos = (pos + direction).clamp(0, self.grid - self.leader_size).to(torch.long)
            x, y = new_pos[0], new_pos[1]

            # Проверка на огонь
            region = self.fire_mask[y:y+self.leader_size, x:x+self.leader_size]
            if region.any():
                self.leader_alive[i] = False
                continue

            # Проверка на выход
            exit_region = self.exit_mask[y:y+self.leader_size, x:x+self.leader_size]
            if exit_region.any():
                self.leader_alive[i] = False
                continue

            self.leader_positions[i] = new_pos

def stack_envs(envs):
    t = lambda attr: torch.stack([getattr(e, attr) for e in envs], 0)
    return (
        t("positions"), t("alive"), t("knows_exit"), t("health"),
        t("size"), t("speed"), t("fire_mask"), t("exit_mask"),
        t("leader_positions"), t("leader_alive")
    )
