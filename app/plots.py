"""Các hàm vẽ biểu đồ cho app demo."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def figure_width(num_syllables):
    return max(6, 0.55 * num_syllables + 2)


def plot_heatmap(matrix, syllables, title, ylabel):
    """Trục x là âm tiết, trục y là các chiều của vector."""
    limit = float(max(abs(matrix.min()), abs(matrix.max()), 1e-6))

    fig, ax = plt.subplots(figsize=(figure_width(len(syllables)), 3.4))
    image = ax.imshow(matrix.T, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(syllables)))
    ax.set_xticklabels(syllables, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_probabilities(syllables, p_begin):
    """Cột xanh: dự đoán B (bắt đầu từ mới). Cột đỏ: dự đoán I (nối từ trước)."""
    colors = ["#2e7d32" if p >= 0.5 else "#c62828" for p in p_begin]

    fig, ax = plt.subplots(figsize=(figure_width(len(syllables)), 3.2))
    ax.bar(range(len(syllables)), p_begin, color=colors)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(syllables)))
    ax.set_xticklabels(syllables, rotation=45, ha="right")
    ax.set_ylabel("P(B)")
    ax.set_title("Xác suất mỗi âm tiết bắt đầu một từ mới")
    fig.tight_layout()
    plt.close(fig)
    return fig
