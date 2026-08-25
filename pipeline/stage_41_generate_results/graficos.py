from pathlib import Path
import torch
import matplotlib.pyplot as plt

# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]  # Mantido o seu padrão original

INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_40_cnn"
MODEL_DIR = INPUT_DIR / "space_frequency_models" / "sub-01"

# --- 2. CRIAÇÃO DA ESTRUTURA (2 linhas, 6 colunas) ---
# Linha 0 (Cima): Loss dos 6 Folds
# Linha 1 (Baixo): Acurácia dos 6 Folds
fig, axs = plt.subplots(3, 6, figsize=(22, 8), sharex=True)
fig.suptitle("Métricas de Treinamento - Todos os 5 Folds", fontsize=16, fontweight='bold')

# --- 3. LOOP PARA CARREGAR E PLOTAR OS 6 FOLDS ---
for fold_idx in range(6):
    fold_num = fold_idx + 1
    file_name = f"space_frequency_fold_{fold_num}.pth"
    file_path = MODEL_DIR / file_name
    
    # Tratamento caso algum fold ainda não tenha sido treinado/salvo
    if not file_path.exists():
        print(f"Aviso: Arquivo do Fold {fold_num} não encontrado. Pulando...")
        continue

    # Carrega o checkpoint de cada fold
    checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
    
    train_loss = checkpoint["train_losses_epoch"]
    test_loss = checkpoint["val_losses_epoch"]
    train_acc = checkpoint["train_accuracies_epoch"]
    test_acc = checkpoint["val_accuracies_epoch"]
    best_epoch = checkpoint["best_epoch"]
    
    
    epochs = range(1, len(train_loss) + 1)
    
    # --- PLOT DA LINHA DE CIMA: LOSS (Linha 0) ---
    ax_loss = axs[0, fold_idx]
    ax_loss.plot(epochs, train_loss, color='blue', label='Train')
    ax_loss.plot(epochs, test_loss, color='red', linestyle='--', label='Validation')
    ax_loss.set_title(f"Fold {fold_num} - Loss  Best Epoch {best_epoch}" , fontsize=11, fontweight='bold')
    ax_loss.grid(True, linestyle=':', alpha=0.5)
    if fold_idx == 0:
        ax_loss.set_ylabel("Loss", fontsize=12)
        ax_loss.legend(loc='upper right') # Legenda apenas no primeiro para não poluir

    # --- PLOT DA LINHA DE BAIXO: ACURÁCIA (Linha 1) ---
    ax_acc = axs[1, fold_idx]
    ax_acc.plot(epochs, train_acc, color='blue', label='Train')
    ax_acc.plot(epochs, test_acc, color='red', linestyle='--', label='Test')
    ax_acc.set_title(f"Fold {fold_num} - Accuracy", fontsize=11, fontweight='bold')
    ax_acc.set_xlabel("Epochs")
    ax_acc.grid(True, linestyle=':', alpha=0.5)
    if fold_idx == 0:
        ax_acc.set_ylabel("Accuracy", fontsize=12)

# --- 4. FINALIZAÇÃO E AJUSTES DE LAYOUT ---
plt.tight_layout()

# Opcional: Se quiser salvar o painel completo de 12 gráficos automaticamente
# fig.savefig(MODEL_DIR / "metricas_todos_folds.png", dpi=300)

plt.show()
