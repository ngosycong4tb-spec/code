import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

from bi_objective_tsp import BiObjectiveTSP, load_tsp_file, generate_random_cities

def run_experiment(coords, pop_size=100, n_gen=200, seed=42, output_dir="images"):
    os.makedirs(output_dir, exist_ok=True)
    n_cities = len(coords)
    print(f"\n{'='*70}")
    print(f"THỰC NGHIỆM TỐI ƯU ĐA MỤC TIÊU: NSGA-II vs MOEA/D")
    print(f"Số lượng thành phố: {n_cities} | Kích thước quần thể: {pop_size} | Số thế hệ: {n_gen}")
    print(f"{'='*70}\n")

    problem = BiObjectiveTSP(coords)

    sampling = PermutationRandomSampling()
    crossover = OrderCrossover(prob=0.9)
    mutation = InversionMutation(prob=0.25)

    print("--> [1/2] Đang chạy thuật toán NSGA-II...")
    algo_nsga2 = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=False
    )

    t0 = time.perf_counter()
    res_nsga2 = minimize(
        problem,
        algo_nsga2,
        ('n_gen', n_gen),
        seed=seed,
        verbose=False
    )
    time_nsga2 = time.perf_counter() - t0
    print(f"    ✓ NSGA-II hoàn thành trong {time_nsga2:.2f} giây.")

    print("--> [2/2] Đang chạy thuật toán MOEA/D...")
    ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=pop_size - 1)
    algo_moead = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=15,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        decomposition=Tchebicheff()
    )

    t0 = time.perf_counter()
    res_moead = minimize(
        problem,
        algo_moead,
        ('n_gen', n_gen),
        seed=seed,
        verbose=False
    )
    time_moead = time.perf_counter() - t0
    print(f"    ✓ MOEA/D hoàn thành trong {time_moead:.2f} giây.")

    nds = NonDominatedSorting()

    front_idx_nsga2 = nds.do(res_nsga2.F, only_non_dominated_front=True)
    pf_nsga2_raw = res_nsga2.F[front_idx_nsga2]
    pop_nsga2_raw = res_nsga2.X[front_idx_nsga2]
    pf_nsga2, u_idx_nsga2 = np.unique(np.round(pf_nsga2_raw, 2), axis=0, return_index=True)
    pop_nsga2 = pop_nsga2_raw[u_idx_nsga2]
    sort_idx2 = np.argsort(pf_nsga2[:, 0])
    pf_nsga2 = pf_nsga2[sort_idx2]
    pop_nsga2 = pop_nsga2[sort_idx2]

    front_idx_moead = nds.do(res_moead.F, only_non_dominated_front=True)
    pf_moead_raw = res_moead.F[front_idx_moead]
    pop_moead_raw = res_moead.X[front_idx_moead]
    pf_moead, u_idx_moead = np.unique(np.round(pf_moead_raw, 2), axis=0, return_index=True)
    pop_moead = pop_moead_raw[u_idx_moead]
    sort_idx_m = np.argsort(pf_moead[:, 0])
    pf_moead = pf_moead[sort_idx_m]
    pop_moead = pop_moead[sort_idx_m]

    combined_F = np.vstack([pf_nsga2, pf_moead])
    ref_front_idx = nds.do(combined_F, only_non_dominated_front=True)
    ref_pf = combined_F[ref_front_idx]
    ref_pf, u_ref = np.unique(np.round(ref_pf, 2), axis=0, return_index=True)
    ref_pf = ref_pf[np.argsort(ref_pf[:, 0])]

    ideal = np.min(combined_F, axis=0)
    nadir = np.max(combined_F, axis=0)
    norm_range = np.where(nadir - ideal == 0, 1.0, nadir - ideal)

    pf_nsga2_norm = (pf_nsga2 - ideal) / norm_range
    pf_moead_norm = (pf_moead - ideal) / norm_range
    ref_pf_norm = (ref_pf - ideal) / norm_range

    ref_point_norm = np.array([1.1, 1.1])
    hv_calc = HV(ref_point=ref_point_norm)

    hv_nsga2 = hv_calc(pf_nsga2_norm)
    hv_moead = hv_calc(pf_moead_norm)

    igd_calc = IGD(ref_pf_norm)
    igd_nsga2 = igd_calc(pf_nsga2_norm)
    igd_moead = igd_calc(pf_moead_norm)

    print("\n" + "="*70)
    print("BẢNG KẾT QUẢ SO SÁNH ĐỊNH LƯỢNG (METRICS COMPARISON)")
    print("="*70)
    header = f"{'Tiêu chí đánh giá':<28} | {'NSGA-II':<18} | {'MOEA/D':<18}"
    print(header)
    print("-" * len(header))
    print(f"{'Số nghiệm Pareto tìm được':<28} | {len(pf_nsga2):<18} | {len(pf_moead):<18}")
    print(f"{'Hypervolume (HV) ↑':<28} | {hv_nsga2:<18.4f} | {hv_moead:<18.4f}")
    print(f"{'IGD (Inverted GD) ↓':<28} | {igd_nsga2:<18.4f} | {igd_moead:<18.4f}")
    print(f"{'Thời gian chạy (giây) ↓':<28} | {time_nsga2:<18.2f} | {time_moead:<18.2f}")
    print(f"{'f1 nhỏ nhất (Min Distance)':<28} | {np.min(pf_nsga2[:, 0]):<18.2f} | {np.min(pf_moead[:, 0]):<18.2f}")
    print(f"{'f2 nhỏ nhất (Min Latency)':<28} | {np.min(pf_nsga2[:, 1]):<18.2f} | {np.min(pf_moead[:, 1]):<18.2f}")
    print("="*70)

    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 1.0

    fig1, ax1 = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    ax1.scatter(res_nsga2.F[:, 0], res_nsga2.F[:, 1], color='#3498db', alpha=0.25, s=25, label='NSGA-II (Tất cả cá thể)')
    ax1.scatter(res_moead.F[:, 0], res_moead.F[:, 1], color='#e67e22', alpha=0.25, s=25, label='MOEA/D (Tất cả cá thể)')
    ax1.plot(pf_nsga2[:, 0], pf_nsga2[:, 1], 'o--', color='#1f77b4', linewidth=2.0, markersize=7, label=f'NSGA-II Pareto (HV={hv_nsga2:.3f})')
    ax1.plot(pf_moead[:, 0], pf_moead[:, 1], 's-', color='#d35400', linewidth=2.0, markersize=7, label=f'MOEA/D Pareto (HV={hv_moead:.3f})')
    ax1.plot(ref_pf[:, 0], ref_pf[:, 1], 'k:', linewidth=1.5, alpha=0.7, label='Reference Pareto Front')

    ax1.set_title(f"So sánh Biên Pareto: NSGA-II vs MOEA/D ({n_cities} Cities)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Mục tiêu 1: Tổng quãng đường chu trình ($f_1$)", fontsize=11, labelpad=8)
    ax1.set_ylabel("Mục tiêu 2: Tổng thời gian chờ đợi ($f_2$)", fontsize=11, labelpad=8)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True, loc='upper right', fontsize=9)
    fig1.tight_layout()
    p1 = os.path.join(output_dir, "tsp_pareto_fronts.png")
    fig1.savefig(p1)
    plt.close(fig1)
    print(f"[OK] Đã lưu đồ thị Biên Pareto: {p1}")

    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(9.5, 4.5), dpi=300)
    algorithms = ['NSGA-II', 'MOEA/D']
    x = np.arange(len(algorithms))
    width = 0.35

    bx1.bar(x - width/2, [hv_nsga2, hv_moead], width, label='HV (càng cao càng tốt)', color='#2ecc71', edgecolor='black')
    bx1.bar(x + width/2, [igd_nsga2, igd_moead], width, label='IGD (càng thấp càng tốt)', color='#e74c3c', edgecolor='black')
    bx1.set_xticks(x)
    bx1.set_xticklabels(algorithms, fontweight='bold')
    bx1.set_title("Độ đo Chất lượng Tập nghiệm (HV & IGD)", fontsize=11, fontweight='bold')
    bx1.set_ylabel("Giá trị chuẩn hóa", fontsize=10)
    bx1.grid(axis='y', linestyle='--', alpha=0.6)
    bx1.legend(loc='upper right', fontsize=9)

    for i in range(len(algorithms)):
        bx1.text(i - width/2, [hv_nsga2, hv_moead][i] + 0.02, f"{[hv_nsga2, hv_moead][i]:.3f}", ha='center', fontsize=9, fontweight='bold')
        bx1.text(i + width/2, [igd_nsga2, igd_moead][i] + 0.02, f"{[igd_nsga2, igd_moead][i]:.3f}", ha='center', fontsize=9, fontweight='bold')

    bx2.bar(x - width/2, [time_nsga2, time_moead], width, label='Thời gian (giây) ↓', color='#3498db', edgecolor='black')
    bx2_twin = bx2.twinx()
    bx2_twin.bar(x + width/2, [len(pf_nsga2), len(pf_moead)], width, label='Số nghiệm Pareto', color='#9b59b6', edgecolor='black')
    bx2.set_xticks(x)
    bx2.set_xticklabels(algorithms, fontweight='bold')
    bx2.set_title("Thời gian & Số nghiệm Pareto", fontsize=11, fontweight='bold')
    bx2.set_ylabel("Thời gian (giây)", fontsize=10, color='#2980b9')
    bx2_twin.set_ylabel("Số lượng nghiệm", fontsize=10, color='#8e44ad')
    bx2.grid(axis='y', linestyle='--', alpha=0.6)

    for i in range(len(algorithms)):
        bx2.text(i - width/2, [time_nsga2, time_moead][i] + 0.1, f"{[time_nsga2, time_moead][i]:.2f}s", ha='center', fontsize=9, fontweight='bold')
        bx2_twin.text(i + width/2, [len(pf_nsga2), len(pf_moead)][i] + 0.5, f"{[len(pf_nsga2), len(pf_moead)][i]}", ha='center', fontsize=9, fontweight='bold')

    fig2.tight_layout()
    p2 = os.path.join(output_dir, "moo_metrics_bar.png")
    fig2.savefig(p2)
    plt.close(fig2)
    print(f"[OK] Đã lưu đồ thị Độ đo so sánh: {p2}")

    fig3, (cx1, cx2) = plt.subplots(1, 2, figsize=(10, 5), dpi=300)

    best_f1_idx = 0
    tour_dist = pop_nsga2[best_f1_idx]
    tour_dist_coords = coords[np.append(tour_dist, tour_dist[0])]
    cx1.plot(tour_dist_coords[:, 0], tour_dist_coords[:, 1], 'o-', color='#1f77b4', markersize=6, linewidth=1.8)
    cx1.scatter(coords[tour_dist[0], 0], coords[tour_dist[0], 1], color='red', s=130, zorder=5, label='Điểm xuất phát (Depot)')
    for idx, (cx, cy) in enumerate(coords):
        cx1.annotate(f"{idx}", (cx+1.2, cy+1.2), fontsize=8, color='#333333')
    cx1.set_title(f"Ưu tiên Quãng đường ($f_1$ Min)\n$f_1={pf_nsga2[best_f1_idx, 0]:.1f}, f_2={pf_nsga2[best_f1_idx, 1]:.1f}$", fontsize=10, fontweight='bold')
    cx1.set_xlabel("Tọa độ X")
    cx1.set_ylabel("Tọa độ Y")
    cx1.grid(True, linestyle=':', alpha=0.6)
    cx1.legend(loc='lower right', fontsize=8)

    best_f2_idx = np.argmin(pf_nsga2[:, 1])
    tour_lat = pop_nsga2[best_f2_idx]
    tour_lat_coords = coords[np.append(tour_lat, tour_lat[0])]
    cx2.plot(tour_lat_coords[:, 0], tour_lat_coords[:, 1], 's-', color='#e74c3c', markersize=6, linewidth=1.8)
    cx2.scatter(coords[tour_lat[0], 0], coords[tour_lat[0], 1], color='red', s=130, zorder=5, label='Điểm xuất phát (Depot)')
    for idx, (cx, cy) in enumerate(coords):
        cx2.annotate(f"{idx}", (cx+1.2, cy+1.2), fontsize=8, color='#333333')
    cx2.set_title(f"Ưu tiên Thời gian chờ ($f_2$ Min)\n$f_1={pf_nsga2[best_f2_idx, 0]:.1f}, f_2={pf_nsga2[best_f2_idx, 1]:.1f}$", fontsize=10, fontweight='bold')
    cx2.set_xlabel("Tọa độ X")
    cx2.set_ylabel("Tọa độ Y")
    cx2.grid(True, linestyle=':', alpha=0.6)
    cx2.legend(loc='lower right', fontsize=8)

    fig3.tight_layout()
    p3 = os.path.join(output_dir, "tsp_tours_tradeoff.png")
    fig3.savefig(p3)
    plt.close(fig3)
    print(f"[OK] Đã lưu đồ thị Minh họa lộ trình Trade-off: {p3}")

    print("\n>>> HOÀN THÀNH TOÀN BỘ THỰC NGHIỆM VÀ XUẤT ĐỒ THỊ THÀNH CÔNG! <<<\n")
    return {
        "hv_nsga2": hv_nsga2,
        "hv_moead": hv_moead,
        "igd_nsga2": igd_nsga2,
        "igd_moead": igd_moead,
        "time_nsga2": time_nsga2,
        "time_moead": time_moead,
        "n_pareto_nsga2": len(pf_nsga2),
        "n_pareto_moead": len(pf_moead)
    }

def main():
    parser = argparse.ArgumentParser(description="So sánh NSGA-II và MOEA/D cho Bi-objective TSP")
    parser.add_argument("file_path", nargs="?", default=None, help="Đường dẫn đến file .tsp hoặc .txt (tùy chọn)")
    parser.add_argument("--n_cities", type=int, default=25, help="Số thành phố ngẫu nhiên nếu không truyền file (mặc định: 25)")
    parser.add_argument("--pop_size", type=int, default=100, help="Kích thước quần thể (mặc định: 100)")
    parser.add_argument("--n_gen", type=int, default=200, help="Số thế hệ tiến hóa (mặc định: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (mặc định: 42)")
    default_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    parser.add_argument("--output_dir", type=str, default=default_img_dir, help="Thư mục lưu đồ thị (mặc định: Week3/images)")

    args = parser.parse_args()

    if args.file_path and os.path.exists(args.file_path):
        print(f"Đang nạp dữ liệu từ file: {args.file_path}")
        coords = load_tsp_file(args.file_path)
    else:
        print(f"Không có file truyền vào hoặc file không tồn tại. Đang tự động sinh {args.n_cities} thành phố ngẫu nhiên...")
        coords = generate_random_cities(n_cities=args.n_cities, seed=args.seed)

    run_experiment(
        coords=coords,
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        seed=args.seed,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
