import os
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # Hoặc dùng PCA nếu tập dữ liệu quá lớn
from sklearn.metrics import silhouette_score, davies_bouldin_score

def extract_pe_matrices_dual(model, evaluator=None):
    """
    Hàm trích xuất nâng cấp: Tự động quét và tìm biến Parameter 'gating' 
    bên trong ProposedPositionalEmbedding.
    """
    state_dict = model.state_dict()
    
    seq_len = None
    if evaluator is not None and hasattr(evaluator, 'dataloader'):
        try:
            X, _, _ = next(iter(evaluator.dataloader))
            if X.ndim == 3:
                seq_len = X.shape[-1] if X.shape[-1] > X.shape[1] else X.shape[1]
            elif X.ndim == 4:
                seq_len = X.shape[-2] if X.shape[-2] > 1 else X.shape[-1]
        except:
            pass
    if seq_len is None or seq_len <= 1:
        seq_len = 50 

    # Mặc định gate = 0.0 nếu không tìm thấy module hoặc biến
    extracted_gate = 0.0

    for name, module in model.named_modules():
        if module.__class__.__name__ == "ProposedPositionalEmbedding":
            device = next(module.relative_compression.parameters()).device
            pe_for_rank = None
            
            # TỰ ĐỘNG TRÍCH XUẤT GATING: Tìm self.gating
            if hasattr(module, 'gating') and getattr(module, 'gating') is not None:
                # Đi qua hàm Sigmoid để lấy giá trị xác suất (0 đến 1) giống trong forward gốc
                extracted_gate = torch.sigmoid(module.gating).detach().cpu().item()
            
            # 1. Trích xuất PE cuối cùng (Cho Stable Rank)
            if hasattr(module, 'dist') and hasattr(module, 'binary_pos') and getattr(module, 'dist') is not None:
                with torch.no_grad():
                    dist_tensor = getattr(module, 'dist').to(device)
                    binary_tensor = getattr(module, 'binary_pos').to(device)
                    
                    rel_out = module.relative_compression(dist_tensor)
                    abs_out = module.absolute_compression(binary_tensor)
                    pe_final = torch.cat([rel_out, abs_out], dim=-1).detach().cpu().squeeze()
                    if pe_final.ndim == 2:
                        pe_for_rank = pe_final

            # 2. Trích xuất thành phần Relative
            if hasattr(module, 'dist') and getattr(module, 'dist') is not None:
                with torch.no_grad():
                    dist_tensor = getattr(module, 'dist').to(device)
                    relative_out = module.relative_compression(dist_tensor).detach().cpu()
            else:
                linear_layer = module.relative_compression[0]
                in_features = linear_layer.in_features
                dummy_dist = torch.zeros(seq_len, in_features, device=device)
                for i in range(seq_len):
                    dummy_dist[i, :min(seq_len, in_features)] = torch.arange(min(seq_len, in_features), device=device)
                
                with torch.no_grad():
                    relative_out = module.relative_compression(dummy_dist).detach().cpu()

            pe_for_plot = relative_out.squeeze()
            
            if pe_for_rank is None:
                pe_for_rank = pe_for_plot

            return pe_for_plot, pe_for_rank, extracted_gate, f"Proposed_PE: {name}"

    # --- FALLBACKS KHÁC ---
    pe_keys = [k for k in state_dict.keys() if any(kw in k.lower() for kw in ['pos_embed', 'pos_encode', 'position', 'pe'])]
    for key in pe_keys:
        matrix = state_dict[key].detach().cpu()
        squeezed = matrix.squeeze()
        if squeezed.ndim == 2:
            return squeezed, squeezed, extracted_gate, f"Static_Key: {key}"

    return None, None, extracted_gate, None


def plot_and_save_results(best_model, best_test_evaluator, problem_name):
    """
    Vẽ ảnh dựa trên thành phần relative (Gated bằng biến self.gating tự động trích xuất), 
    tính Stable Rank, trực quan hóa không gian Latent phối hợp PCA + t-SNE, và trả về bộ 3 chỉ số.
    """
    output_dir = "Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    stable_rank_val = None
    sh_score = 0.0
    db_index = 0.0
    
    # Lấy dữ liệu PE và giá trị gate tự động từ mô hình
    pe_plot_tensor, pe_rank_tensor, gate_val, pe_info = extract_pe_matrices_dual(best_model, best_test_evaluator)
    
    # Tính hệ số (1 - gate)
    gate_coef = 1.0 - gate_val

    if pe_plot_tensor is not None and pe_rank_tensor is not None:
        # 1. --- TÍNH STABLE RANK ---
        pe_rank_np = pe_rank_tensor.to(torch.float32).numpy()
        _, U_rank, _ = np.linalg.svd(pe_rank_np, full_matrices=False)
        frobenius_sq = np.sum(U_rank ** 2)
        operator_sq = np.max(U_rank) ** 2
        stable_rank_val = frobenius_sq / operator_sq if operator_sq > 0 else 0

        # 2. --- ĐỒ THỊ 2: PE Similarity vs Distance (Gated với 1 - Gate) ---
        L = pe_plot_tensor.shape[0]
        if pe_plot_tensor.ndim == 2 and pe_plot_tensor.shape[0] != pe_plot_tensor.shape[1]:
            sim_matrix = pe_plot_tensor @ pe_plot_tensor.T
            L = sim_matrix.shape[0]
        else:
            sim_matrix = pe_plot_tensor

        Ks = torch.arange(0, L)
        avg_sim = []
        for k in Ks:
            if k == 0:
                vals = sim_matrix.diagonal(0)
            else:
                vals = torch.cat([sim_matrix.diagonal(k.item()), sim_matrix.diagonal(-k.item())])
            
            # Nhân hệ số (1 - gate) tự động lấy từ mô hình
            scaled_sim = vals.mean().item() * gate_coef
            avg_sim.append(scaled_sim)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(Ks.numpy(), avg_sim, marker='o', markersize=2, linestyle='-', color='crimson')
        ax.set_xlabel("K = |i - j| (Temporal Distance)")
        ax.set_ylabel("Gated Similarity Bias (1 - Gate) * Sim")
        ax.set_title(f"PE Gated Similarity vs Distance - {problem_name}\n(Gating: {gate_val:.4f}, 1-Gating: {gate_coef:.4f})", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{problem_name}_PE_distance_similarity.png"), dpi=300)
        plt.close(fig)
    else:
        print(f"\n[Lưu ý] Không trích xuất được cấu trúc PE từ {problem_name}.")

    # 3. --- ĐỒ THỊ 3: Trực quan hóa Latent Space (PCA + t-SNE) ---
    best_test_evaluator.model.eval()
    all_features, all_labels = [], []
    target_layer = best_test_evaluator.model.flatten
    
    latent_storage = []
    def hook_fn(module, input, output):
        latent_storage.append(output.detach().cpu())

    hook_handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        for X_batch, y_batch, *_ in best_test_evaluator.dataloader:
            X_batch = X_batch.to(best_test_evaluator.device)
            _ = best_test_evaluator.model(X_batch)
            all_features.append(latent_storage[-1].numpy())
            all_labels.append(y_batch.numpy())
            
    hook_handle.remove()
        
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    try:
        sh_score = silhouette_score(all_features, all_labels)
        db_index = davies_bouldin_score(all_features, all_labels)
        title_score_str = f" (Silhouette Score: {sh_score:.3f})"
    except:
        title_score_str = ""

    if all_features.shape[1] > 50:
        pca = PCA(n_components=min(50, all_features.shape[0]), random_state=42)
        features_for_tsne = pca.fit_transform(all_features)
    else:
        features_for_tsne = all_features

    n_samples = features_for_tsne.shape[0]
    perplexity = min(30, max(5, n_samples // 3)) 
    
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
        latent_2d = tsne.fit_transform(features_for_tsne)
    except TypeError:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        latent_2d = tsne.fit_transform(features_for_tsne)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(all_labels)
    palette = sns.color_palette("hls", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        idx = (all_labels == label)
        plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], label=f'Class {label}', 
                    alpha=0.8, edgecolors='w', s=45, color=palette[i])
        
    plt.title(f'Latent Space Clustering via PCA + t-SNE - {problem_name}{title_score_str}', fontweight='bold', pad=15)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="True Labels", ncol=2 if len(unique_labels) > 12 else 1)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{problem_name}_latent_representation.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return stable_rank_val, sh_score, db_index
def plot_stable_rank_summary(stable_rank_dict):
    """Hàm mới: Nhận vào dictionary kết quả và vẽ biểu đồ cột Stable Rank tổng quan"""
    if not stable_rank_dict:
        print("[Lưu ý] Dictionary trống, không có dữ liệu Stable Rank để vẽ.")
        return

    output_dir = "Visualizations"
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(stable_rank_dict.keys())
    ranks = list(stable_rank_dict.values())

    plt.figure(figsize=(max(10, len(datasets) * 0.8), 6))
    
    # Vẽ biểu đồ cột nền nã với bảng màu chuyên nghiệp
    colors = sns.color_palette("muted", len(datasets))
    bars = plt.bar(datasets, ranks, color=colors, edgecolor='black', width=0.55)
    
    # Thêm giá trị số cụ thể lên đầu mỗi cột
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + max(ranks)*0.01, f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.title("PE Stable Rank Comparison Across All Datasets", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Datasets", fontsize=12, fontweight='bold')
    plt.ylabel("Stable Rank Value", fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(0, max(ranks) * 1.15)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "All_Datasets_PE_Stable_Rank.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n>>>> ĐÃ LƯU BIỂU ĐỒ TỔNG HỢP STABLE RANK TẠI: {save_path} <<<<")
    plt.close()

def plot_classification_metrics_summary(silhouette_dict, dbi_dict):
    """Hàm mới: Vẽ và lưu tách biệt 2 đồ thị cột độc lập cho Silhouette và DBI"""
    if not silhouette_dict or not dbi_dict:
        print("[Lưu ý] Không có dữ liệu Metric phân lớp để vẽ.")
        return

    output_dir = "Visualizations"
    os.makedirs(output_dir, exist_ok=True)

    datasets = list(silhouette_dict.keys())
    silhouettes = [silhouette_dict[d] for d in datasets]
    db_indices = [dbi_dict[d] for d in datasets]
    
    # Cấu hình chiều rộng linh hoạt theo số lượng dataset
    fig_width = max(10, len(datasets) * 0.8)

    # -----------------------------------------------------------------
    # ĐỒ THỊ 1: Tách riêng Silhouette Score (Càng cao càng tốt)
    # -----------------------------------------------------------------
    plt.figure(figsize=(fig_width, 5))
    # Sử dụng tông màu xanh dương học thuật (Academic Blue)
    bars_sil = plt.bar(datasets, silhouettes, color='#1f77b4', edgecolor='black', width=0.5)
    
    # Ghi số cụ thể lên đầu cột
    for bar in bars_sil:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + max(silhouettes)*0.01 if yval >= 0 else yval - 0.05, 
                 f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9, color='#1f77b4')

    plt.title("Latent Space Clustering Quality: Silhouette Score Comparison", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Datasets", fontsize=11, fontweight='bold')
    plt.ylabel("Silhouette Score (Higher is Better)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(min(-0.1, min(silhouettes) - 0.05), 1.05)
    plt.tight_layout()
    
    sil_path = os.path.join(output_dir, "All_Datasets_Silhouette_Score.png")
    plt.savefig(sil_path, dpi=300)
    plt.close()
    print(f"\n>>>> ĐÃ LƯU ĐỒ THỊ SILHOUETTE RỜI TẠI: {sil_path} <<<<")


    # -----------------------------------------------------------------
    # ĐỒ THỊ 2: Tách riêng Davies-Bouldin Index (Càng thấp càng tốt)
    # -----------------------------------------------------------------
    plt.figure(figsize=(fig_width, 5))
    # Sử dụng tông màu cam đất tiêu chuẩn (Muted Orange)
    bars_db = plt.bar(datasets, db_indices, color='#ff7f0e', edgecolor='black', width=0.5)
    
    # Ghi số cụ thể lên đầu cột
    for bar in bars_db:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + max(db_indices)*0.01, 
                 f"{yval:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9, color='#d62728')

    plt.title("Latent Space Clustering Quality: Davies-Bouldin Index Comparison", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Datasets", fontsize=11, fontweight='bold')
    plt.ylabel("Davies-Bouldin Index (Lower is Better)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.ylim(0, max(db_indices) * 1.15 if max(db_indices) > 0 else 2.0)
    plt.tight_layout()
    
    db_path = os.path.join(output_dir, "All_Datasets_Davies_Bouldin_Index.png")
    plt.savefig(db_path, dpi=300)
    plt.close()
    print(f">>>> ĐÃ LƯU ĐỒ THỊ DBI RỜI TẠI: {db_path} <<<<")