from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]  # Mantido o seu padrão original

INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_40_cnn"

subjects = [f"sub-{i:02d}" for i in range(1, 11)]

def individual_subject_plots(branch = "space_frequency", subjects = subjects ):

    for subject in subjects:

        MODEL_DIR = INPUT_DIR / f"{branch}_models" / f"{subject}"
                    
        match branch:
            case "space_frequency":
                title = " Space x Space x Frequency - Channel: Time" 
            case _:
               raise ValueError(f"Invalid Branch {branch}")


        fold_files = sorted(MODEL_DIR.glob("space_frequency_fold_*.pth"))
        n_folds = len(fold_files)

       # --- 2. CRIAÇÃO DA ESTRUTURA (3 linhas,  n_folds colunas) ---
        fig, axs = plt.subplots(3, n_folds + 1, figsize=(22, 10))

        fig.suptitle(f"{title} / {subject.capitalize()}", fontsize=12, fontweight='bold')

        # ----------------------------------------
        # LOOP PARA CARREGAR E PLOTAR OS FOLDS
        # ----------------------------------------

        for fold_idx in range(n_folds):
            fold_num = fold_idx + 1
            file_name = f"{branch}_fold_{fold_num}.pth"
            file_path = MODEL_DIR / file_name
            
            if not file_path.exists():
                print(f"Aviso: Arquivo do Fold {fold_num} não encontrado. Pulando...")
                continue

            checkpoint = torch.load(file_path, map_location="cpu", weights_only=False)
            
            train_losses_epoch = checkpoint["train_losses_epoch"]
            val_losses_epoch = checkpoint["val_losses_epoch"]
            train_accuracies_epoch = checkpoint["train_accuracies_epoch"]
            val_accuracies_epoch = checkpoint["val_accuracies_epoch"]
            min_val_loss_epoch = checkpoint["min_val_loss_epoch"]
            fold_accuracies = checkpoint["minimum_loss_model_test_results"]["fold_accuracies"]
            fold_losses = checkpoint["minimum_loss_model_test_results"]["fold_losses"]
            all_labels = checkpoint["minimum_loss_model_test_results"]["all_labels"]
            all_predictions = checkpoint["minimum_loss_model_test_results"]["all_predictions"]
            
            epochs = range(1, len(train_losses_epoch) + 1)
            
            # ---------------------------------------
            # PLOT DA LINHA DE CIMA: LOSS (Linha 0)
            # ---------------------------------------

            ax_loss = axs[0, fold_idx]
            ax_loss.plot(epochs, train_losses_epoch, color='blue', label='Train')
            ax_loss.plot(epochs, val_losses_epoch, color='red', label='Validation')
            
            # Adiciona a linha vertical na melhor época para o gráfico de Loss
            ax_loss.axvline(x=min_val_loss_epoch + 1 , color='forestgreen', linestyle=':', linewidth=2, label='Min Val Loss')
            
            ax_loss.set_title(f"Fold {fold_num} - Loss", fontsize=11, fontweight='bold')
            ax_loss.grid(True, linestyle=':', alpha=0.5)
            ax_loss.set_xticklabels([]) 
            
            if fold_idx == 0:
                ax_loss.set_ylabel("Loss", fontsize=12)
                ax_loss.legend(loc='upper left')

            # -------------------------------------------
            # PLOT DA LINHA DO MEIO: ACURÁCIA (Linha 1)
            # -------------------------------------------

            ax_acc = axs[1, fold_idx]
            ax_acc.plot(epochs, train_accuracies_epoch, color='blue', label='Train')
            ax_acc.plot(epochs, val_accuracies_epoch, color='red', label='Validation')
            
            # Adiciona a linha vertical na melhor época para o gráfico de Acurácia
            ax_acc.axvline(x=min_val_loss_epoch + 1, color='forestgreen', linestyle=':', linewidth=2, label='Min Val Loss')
            
            ax_acc.set_title(f"Fold {fold_num} - Accuracy", fontsize=11, fontweight='bold')
            ax_acc.set_xlabel("Epochs")
            ax_acc.grid(True, linestyle=':', alpha=0.5)
            
            if fold_idx == 0:
                ax_acc.set_ylabel("Accuracy", fontsize=12)
                ax_acc.legend(loc='upper left') 

            # ---------------------------------------------------
            # CONSTRUÇÃO DA MINI-TABELA INDIVIDUAL: (Linha 2)
            # ---------------------------------------------------

            ax_table = axs[2, fold_idx]
            ax_table.axis('off')
            
            val_loss_best_epoch = val_losses_epoch[min_val_loss_epoch]
            val_accuracy_best_epoch = val_accuracies_epoch[min_val_loss_epoch]
            test_loss = checkpoint["minimum_loss_model_test_results"]["test_loss"]
            test_accuracy = checkpoint["minimum_loss_model_test_results"]["test_accuracy"]
        
            cell_text = [
                [str(min_val_loss_epoch + 1)],
                [f"{val_loss_best_epoch:.4f}"],
                [f"{val_accuracy_best_epoch:.4f}"],
                [f"{test_loss:.4f}"],
                [f"{test_accuracy:.4f}"],
                [np.bincount(all_labels, minlength=4)],
                [np.bincount(all_predictions, minlength=4)],
            ]
            
            row_labels = ["Best Epoch", "Val Loss", "Val Acc", "Test Loss", "Test Acc", "All Labels", "All Predictions"] if fold_idx == 0 else None
            
            mini_table = ax_table.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=["Min Val Loss"],
                loc='center',
                cellLoc='center'
            )
            
            mini_table.set_fontsize(10)
            mini_table.scale(1, 1.1)


        # ------------------------------------------
        #  TABELA DE RESULTADO FINAL DE TESTE
        # ------------------------------------------

        file_name = fold_files[-1]
        # file_name = f"space_frequency_fold_5.pth"
        # file_path = MODEL_DIR / file_name

        ax_table = axs[2, n_folds]
        ax_table.axis('off')
        cell_text = []


        cell_text = [ [f"{fold_accuracies[fold_idx]:.4f}"] for fold_idx in range(n_folds)]


        fold_accuracies_mean = f"{fold_accuracies.mean():.4f}"
        fold_accuracies_std = f"{fold_accuracies.std():.4f}"

        cell_text.append([fold_accuracies_mean])
        cell_text.append([fold_accuracies_std])

        row_labels = [f"Fold {fold_idx+1}" for fold_idx in range(n_folds)]
        row_labels.append("Mean")
        row_labels.append("Std")

        mini_table = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=[f"Test - Min Val Loss"],
        loc='center',
        cellLoc='center'
        )

        mini_table.set_fontsize(10)
        mini_table.scale(1, 1.5)

        # 1. Modificar os valores numéricos (Coluna 0 do conteúdo)
        # Penúltima linha = Mean, Última linha = Std (A linha 0 é o cabeçalho)

        num_rows = len(cell_text)
        mini_table[num_rows-1, 0].get_text().set_weight('bold')
        mini_table[num_rows, 0].get_text().set_weight('bold')

        # 2. Modificar os rótulos laterais (Row Labels ficam na Coluna -1)
        mini_table[num_rows-1, -1].get_text().set_weight('bold')
        mini_table[num_rows, -1].get_text().set_weight('bold')

        #Opcional: Mudar a cor do texto ou fundo delas para destacar ainda mais
        mini_table[num_rows-1, 0].set_facecolor('#e6f2ff')
        mini_table[num_rows, 0].set_facecolor('#e6f2ff')

        # Ajusta o tamanho da fonte e escala para caber confortavelmente
        mini_table.set_fontsize(10)
        mini_table.scale(1, 1)

        # ax_table = axs[0, 5]
        # ax_table.axis('off')

        # ax_table = axs[1, 5]
        # ax_table.axis('off')

        # --- 4. FINALIZAÇÃO E AJUSTES DE LAYOUT ---
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    individual_subject_plots(branch="space_frequency", subjects=["sub-02",])