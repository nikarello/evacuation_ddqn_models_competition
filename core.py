
import torch
import torch.nn.functional as F

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Reward Settings ===
STEP_PENALTY     = -1.0     # Штраф за каждый шаг
FIRE_PENALTY     = -50.0    # Штраф за попадание в огонь
EXIT_REWARD      = +300.0   # Награда за эвакуацию
DEATH_PENALTY    = -100.0   # Штраф за смерть в огне
MOVE_TOWARD_EXIT_REWARD = 1  # Бонус за движение к выходу
NEAR_LEADER_REWARD = 2.0   # Бонус за близость к лидеру
REWARD_SCALE = 0.01

# Направления движения: up, down, left, right
DIRS = torch.tensor([[0, -1],
                     [0,  1],
                     [-1, 0],
                     [ 1, 0]], dtype=torch.long, device=device)

# Сдвиги для агентов 1x1, 2x2, 3x3
OFFS = {
    sz: torch.stack(
            torch.meshgrid(
                torch.arange(sz, device=device),
                torch.arange(sz, device=device),
                indexing="ij"
            ), dim=-1
        ).view(-1, 2)
    for sz in (1, 2, 3)
}

# Ядро для распространения огня
FIRE_KERNEL = torch.tensor(
    [[0,1,0],
     [1,0,1],
     [0,1,0]],
    device=device,
    dtype=torch.float32
)[None,None]

def batched_update_fire_exit(fire_mask, exit_mask, p_spread=0.3):
    inp    = fire_mask.unsqueeze(1)
    spread = F.conv2d(inp, FIRE_KERNEL, padding=1).squeeze(1)
    prob   = (spread>0) & (fire_mask==0) & (~exit_mask)
    newf   = (torch.rand_like(spread) < p_spread) & prob
    fire_mask = torch.clamp(fire_mask + newf.float(), max=1.0)
    return fire_mask, exit_mask

def batched_update_agents(positions, size, speed, knows_exit, alive, grid_size,
                           leader_positions=None, leader_alive=None, leader_size=3):
    B, N, _ = positions.shape

    positions  = positions.clone()
    size       = size.clone()
    speed      = speed.clone()
    knows_exit = knows_exit.clone()

    positions[~alive]  = -1
    size     [~alive]  = 0
    speed    [~alive]  = 0
    knows_exit[~alive] = False

    H = W = grid_size
    ag_map  = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    sz_map  = torch.zeros_like(ag_map)
    sp_map  = torch.zeros_like(ag_map)
    inf_map = torch.zeros_like(ag_map)

    size_cells = (size * 5).round().long()

    for sz in (1, 2, 3):
        mask = (size_cells == sz)
        if not mask.any():
            continue

        b_idx, a_idx = torch.nonzero(mask, as_tuple=True)
        if b_idx.numel() == 0:
            continue

        pivots = positions[b_idx, a_idx]
        offs = OFFS[sz]
        cells = pivots.unsqueeze(1) + offs.unsqueeze(0)
        cells = cells.view(-1, 2)
        xs, ys = cells[:, 0], cells[:, 1]

        K = sz * sz
        b_rep = b_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        a_rep = a_idx.unsqueeze(1).expand(-1, K).reshape(-1)

        ag_map[b_rep, ys, xs] = 1.0
        sz_map[b_rep, ys, xs] = sz / 5.0

        sp_vals = speed[b_idx, a_idx]
        sp_rep = sp_vals.unsqueeze(1).expand(-1, K).reshape(-1)
        sp_map[b_rep, ys, xs] = sp_rep

        inf_vals = knows_exit[b_idx, a_idx].float()
        inf_rep = inf_vals.unsqueeze(1).expand(-1, K).reshape(-1)
        inf_map[b_rep, ys, xs] = inf_rep

    leader_map = torch.zeros_like(ag_map)
    if leader_positions is not None:
        B, L, _ = leader_positions.shape
        for b in range(B):
            for l in range(L):
                if leader_alive[b, l]:
                    pos = leader_positions[b, l]
                    cells = pos.unsqueeze(0) + OFFS[leader_size]
                    ys, xs = cells[:, 1], cells[:, 0]
                    leader_map[b, ys, xs] = 1.0

    return ag_map, sz_map, sp_map, inf_map, leader_map


def batched_get_views(
    leader_map, agent_map, fire_mask, exit_mask,
    size_map, speed_map, info_map, positions, view_size
):
    """
    ⚡ Быстрая версия получения обзоров агентов через векторизованный gather.
    """
    B, N, _ = positions.shape
    C = 8
    V = view_size
    half = V // 2

    # Объединяем все карты в один тензор
    maps = torch.stack([
        leader_map, agent_map, fire_mask, exit_mask.float(),
        size_map, speed_map, info_map,
        torch.zeros_like(fire_mask)  # border_mask — будет заполнена вручную
    ], dim=1)  # [B, C, H, W]

    H, W = fire_mask.shape[1:]
    pad = (half, half, half, half)  # L, R, T, B
    maps = F.pad(maps, pad)

    # Заполняем border mask
    maps[:, -1].fill_(0)
    maps[:, -1, :half].fill_(1)
    maps[:, -1, -(half):].fill_(1)
    maps[:, -1, :, :half].fill_(1)
    maps[:, -1, :, -(half):].fill_(1)

    # Смещение позиций агентов из-за паддинга
    pos = positions + half  # [B, N, 2]
    y, x = pos.unbind(-1)   # [B, N]

    # Координатная сетка обзора
    d = torch.arange(-half, half + 1, device=positions.device)
    dy, dx = torch.meshgrid(d, d, indexing="ij")  # [V, V]
    dy = dy.view(1, 1, V, V)
    dx = dx.view(1, 1, V, V)

    # Смещённые координаты для каждого агента
    ys = (y.unsqueeze(-1).unsqueeze(-1) + dy).clamp_(0, H + 2 * half - 1)
    xs = (x.unsqueeze(-1).unsqueeze(-1) + dx).clamp_(0, W + 2 * half - 1)

    # Преобразуем в линейные индексы
    lin_idx = ys * (W + 2 * half) + xs  # [B, N, V, V]
    lin_idx = lin_idx.view(B, N, -1)   # [B, N, V*V]

    # Преобразуем карты: [B, C, H, W] → [B, H*W, C]
    maps_flat = maps.view(B, C, -1).permute(0, 2, 1)

    # Собираем обзоры с помощью gather
    views = torch.gather(
        maps_flat.unsqueeze(1).expand(-1, N, -1, -1),   # [B, N, H*W, C]
        2,                                              # индексируем по линейному индексу
        lin_idx.unsqueeze(-1).expand(-1, -1, -1, C)     # [B, N, V*V, C]
    )

    # Финальное преобразование: [B, N, V*V, C] → [B, N, C, V, V]
    views = views.view(B, N, V, V, C).permute(0, 1, 4, 2, 3).contiguous()
    return views







def batched_step(positions, actions, size, speed, fire_mask, exit_mask, health, leader_pos, leader_alive, leader_size):
    B, N, _ = positions.shape
    sp_cells = speed.long().unsqueeze(-1)
    delta    = DIRS[actions] * sp_cells

    # 1. Предложенные новые позиции
    prop = positions + delta
    prop = prop.clamp(min=0)
    maxc = (fire_mask.size(1) - (size * 5).round().long()).unsqueeze(-1)
    prop = torch.min(prop, maxc)

    alive = torch.ones_like(size, dtype=torch.bool)  

    # 2. Проверка коллизий
    grid_size = fire_mask.size(1)
    new_pos, coll_mask = _reject_collisions(prop, positions, size, speed, grid_size, alive)

    # 3. Начисление базовых наград
    rewards = torch.full((B, N), STEP_PENALTY, device=device)

    ys, xs = new_pos[..., 1], new_pos[..., 0]
    idx_flat = ys * fire_mask.size(2) + xs
    flat_fire = fire_mask.view(B, -1)
    hits = torch.gather(flat_fire, 1, idx_flat) > 0.5

    exs = torch.zeros((B, N), dtype=torch.bool, device=device)

    for sz in (1, 2, 3):
        mask_sz = ((size * 5).round().long() == sz)
        if not mask_sz.any():
            continue
        b_idx, a_idx = torch.nonzero(mask_sz, as_tuple=True)
        pivots = new_pos[b_idx, a_idx]
        cells  = pivots.unsqueeze(1) + OFFS[sz].unsqueeze(0)
        cells  = cells.view(-1, 2)
        xs_c, ys_c = cells[:, 0], cells[:, 1]
        K = sz * sz
        b_rep = b_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        a_rep = a_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        ex_cells = exit_mask[b_rep, ys_c, xs_c]
        exs.index_put_((b_rep, a_rep), ex_cells.bool(), accumulate=True)

    # === Бонус за движение к выходу ===
    exit_centers = exit_mask.float().nonzero().view(-1, 3)[:, 1:].float().mean(dim=0)  # [2]
    vec_to_exit = exit_centers.view(1, 1, 2) - positions.float()
    dist_to_exit = torch.norm(vec_to_exit, dim=2).clamp(min=1.0)
    vec_to_exit = F.normalize(vec_to_exit, dim=2)
    move_dir = (prop - positions).float()
    move_dir = F.normalize(move_dir, dim=2)
    alignment = (vec_to_exit * move_dir).sum(dim=2)
    toward_exit = alignment > 0.7
    bonus_weight = MOVE_TOWARD_EXIT_REWARD / dist_to_exit
    rewards += toward_exit.float() * bonus_weight

    # === Бонус за близость к лидеру ===
    if leader_pos is not None and leader_alive is not None and leader_pos.shape[1] > 0:
        leader_centers = leader_pos.float() + leader_size / 2  # [B, L, 2]
        agent_pos = positions.float().unsqueeze(2)  # [B, N, 1, 2]
        leader_pos_alive = leader_centers * leader_alive.unsqueeze(-1)  # [B, L, 2]

        diff = agent_pos - leader_pos_alive.unsqueeze(1)  # [B, N, L, 2]
        dist = torch.norm(diff, dim=3)  # [B, N, L]
        dist = dist + (~leader_alive.unsqueeze(1)).float() * 1e6  # маскируем мёртвых

        min_dists = dist.min(dim=2).values  # [B, N]
        visible_radius = fire_mask.size(1) // 2 if fire_mask.size(1) == fire_mask.size(2) else min(fire_mask.size(1), fire_mask.size(2)) // 2
        view_radius = visible_radius if visible_radius >= 1 else 1  # подстраховка
        rewards += (min_dists <= view_radius).float() * NEAR_LEADER_REWARD

    # 4. Награды / Штрафы
    rewards = rewards + hits.float() * FIRE_PENALTY + exs.float() * EXIT_REWARD
    health2 = health - hits.float() * 25.0
    died = hits & (health2 <= 0)
    rewards = rewards + died.float() * DEATH_PENALTY
    rewards *= REWARD_SCALE

    # 5. Обновление состояний
    health = health2.clamp(min=0.0)
    alive = ~(died | exs)
    f2, _ = batched_update_fire_exit(fire_mask, exit_mask)
    dones = ~alive.any(dim=1)

    return new_pos, rewards, dones, alive, health, f2, exit_mask, died, exs


# ─── helpers ──────────────────────────────────────────────────────────
def _reject_collisions(prop, positions, size, speed, grid_size, alive):
    """Запретить ходы, если предложенная позиция пересекается с занятыми."""
    occ0, _, _, _, _ = batched_update_agents(
        positions, size, speed,
        knows_exit=torch.zeros_like(size, dtype=torch.bool, device=device),
        alive=alive, 
        grid_size=grid_size
    )

    B, N = size.shape
    coll = torch.zeros((B, N), dtype=torch.bool, device=device)

    for sz_val in (1, 2, 3):
        mask_sz = ((size * 5).round().long() == sz_val)
        if not mask_sz.any(): continue
        offs = OFFS[sz_val]
        b_idx, a_idx = torch.nonzero(mask_sz, as_tuple=True)
        pivots = prop[b_idx, a_idx]
        cells  = pivots.unsqueeze(1) + offs
        cells  = cells.view(-1, 2)
        xs, ys = cells[:, 0], cells[:, 1]
        K = sz_val * sz_val
        b_r = b_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        a_r = a_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        # Убираем собственные клетки агента из карты занятости (occ0)
        # 1. Рассчитываем старые клетки (до движения)
        old_pivots = positions[b_idx, a_idx]
        old_cells = old_pivots.unsqueeze(1) + offs
        old_cells = old_cells.view(-1, 2)
        xs_old, ys_old = old_cells[:, 0], old_cells[:, 1]
        occ0[b_idx.repeat_interleave(K), ys_old, xs_old] = 0.0

        # 2. Проверяем коллизии без собственных клеток
        hit = occ0[b_r, ys, xs] > 0.5
        coll.index_put_((b_r, a_r), hit, accumulate=True)

    new_pos = torch.where(coll.unsqueeze(-1), positions, prop)
    return new_pos, coll
