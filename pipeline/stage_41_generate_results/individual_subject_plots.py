from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np


# --- 1. CONFIGURAÇÃO DE CAMINHOS ---
current_file = Path(__file__).resolve()
PROJECT_ROOT = current_file.parents[2]  # Mantido o seu padrão original

INPUT_DIR = PROJECT_ROOT / "processed_data" / "stage_40_cnn"

OUTPUT_DIR = INPUT_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FOLDS_DIR = INPUT_DIR / "fold_indices"


DEFAULT_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]

def individual_subject_plots(branch = "space1_space2_frequency_time", subjects = None ):

    # =================================
    # CHECAGEM DA VARIÁVEL SUBJECT
    # =================================
    if isinstance(subjects, str):  #Se o usuário passou uma única string, transforma em lista com um elemento
        subjects = [subjects]
    
    elif subjects is None: # Caso o usuário não passe nada, usa o comportamento padrão
        subjects = DEFAULT_SUBJECTS 

    # ============================
    # LOOP PARA CADA SUJEITO
    # ============================
    for subject in subjects:

        MODEL_DIR = INPUT_DIR / f"{branch}_branch" / f"{subject}"
                    
        match branch:
            case "space1_space2_frequency_time":

                title = " Space1 x Space2 x Frequency - Channel: Time" 
                fold_files = sorted(MODEL_DIR.glob("space1_space2_frequency_time_fold_*.pth"))

            case "space1_space2_time_frequency":

                title = " Space1 x Space2 x Time - Channel: Frequency" 
                fold_files = sorted(MODEL_DIR.glob("space1_space2_time_frequency_fold_*.pth"))

            case "time_frequency_space2_space1":
        
                title = " Time x Frequency x Space2 - Channel: Space1" 
                fold_files = sorted(MODEL_DIR.glob("time_frequency_space2_space1_fold_*.pth"))

            case "time_frequency_space1_space2":
        
                title = " Time x Frequency x Space1 - Channel: Space2" 
                fold_files = sorted(MODEL_DIR.glob("time_frequency_space1_space2_fold_*.pth"))

            case _:
               raise ValueError(f"Invalid Branch {branch}")



        n_folds = len(fold_files)

       # --- 2. CRIAÇÃO DA ESTRUTURA (3 linhas,  n_folds colunas) ---
        fig, axs = plt.subplots(3, n_folds + 1, figsize=(22, 10))

        fig.suptitle(f"{title} / {subject.capitalize()}", fontsize=12, fontweight='bold')

        # ========================================
        # LOOP PARA CARREGAR E PLOTAR OS FOLDS
        # ========================================

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
            max_val_accuracy_epoch = checkpoint["max_val_accuracy_epoch"]

            fold_accuracies_ml = checkpoint["minimum_loss_model_test_results"]["fold_accuracies"]
            fold_accuracies_ma = checkpoint["maximum_accuracy_model_test_results"]["fold_accuracies"]

            fold_losses = checkpoint["minimum_loss_model_test_results"]["fold_losses"]
            fold_losses = checkpoint["maximum_accuracy_model_test_results"]["fold_losses"]


            all_labels_ml = checkpoint["minimum_loss_model_test_results"]["all_labels"]
            all_labels_ma = checkpoint["maximum_accuracy_model_test_results"]["all_labels"]


            all_predictions_ml = checkpoint["minimum_loss_model_test_results"]["all_predictions"]
            all_predictions_ma = checkpoint["maximum_accuracy_model_test_results"]["all_predictions"]
            
            epochs = range(1, len(train_losses_epoch) + 1)
            
            # ---------------------------------------
            # PLOT DA LINHA DE CIMA: LOSS (Linha 0)
            # ---------------------------------------

            ax_loss = axs[0, fold_idx]
            ax_loss.plot(epochs, train_losses_epoch, color='sandybrown', label='Train')
            ax_loss.plot(epochs, val_losses_epoch, color='royalblue', label='Validation')
            
            # Adiciona a linha vertical na melhor época para o gráfico de Loss
            ax_loss.axvline(x=min_val_loss_epoch + 1 , color='forestgreen', linestyle=':', linewidth=2, label='Min Val Loss')
            ax_loss.axvline(x=max_val_accuracy_epoch + 1 , color='violet', linestyle=':', linewidth=2, label='Max Val Acc')

            
            ax_loss.set_title(f"Fold {fold_num} - Loss", fontsize=11, fontweight='bold')
            ax_loss.grid(True, linestyle=':', alpha=0.5)
            ax_loss.set_xticklabels([]) 
            
            if fold_idx == 0:
                ax_loss.set_ylabel("Loss", fontsize=12)
                ax_loss.legend(loc='best')

            # -------------------------------------------
            # PLOT DA LINHA DO MEIO: ACURÁCIA (Linha 1)
            # -------------------------------------------

            ax_acc = axs[1, fold_idx]
            ax_acc.plot(epochs, train_accuracies_epoch, color='sandybrown', label='Train')
            ax_acc.plot(epochs, val_accuracies_epoch, color='royalblue', label='Validation')
            
            # Adiciona a linha vertical na melhor época para o gráfico de Acurácia
            ax_acc.axvline(x=min_val_loss_epoch + 1, color='forestgreen', linestyle=':', linewidth=2, label='Min Val Loss')
            ax_acc.axvline(x=max_val_accuracy_epoch + 1 , color='violet', linestyle=':', linewidth=2, label='Max Val Acc')

            
            ax_acc.set_title(f"Fold {fold_num} - Accuracy", fontsize=11, fontweight='bold')
            ax_acc.set_xlabel("Epochs")
            ax_acc.grid(True, linestyle=':', alpha=0.5)
            
            if fold_idx == 0:
                ax_acc.set_ylabel("Accuracy", fontsize=12)
                ax_acc.legend(loc='best') 

            # ===================================================
            # CONSTRUÇÃO DA MINI-TABELA INDIVIDUAL: (Linha 2)
            # ===================================================

            ax_table = axs[2, fold_idx]
            ax_table.axis('off')

            val_loss_best_epoch_ml = val_losses_epoch[min_val_loss_epoch]
            val_loss_best_epoch_ma = val_losses_epoch[max_val_accuracy_epoch]

            val_accuracy_best_epoch_ml = val_accuracies_epoch[min_val_loss_epoch]
            val_accuracy_best_epoch_ma = val_accuracies_epoch[max_val_accuracy_epoch]

            test_loss_ml = checkpoint["minimum_loss_model_test_results"]["test_loss"]
            test_loss_ma = checkpoint["maximum_accuracy_model_test_results"]["test_loss"]

            test_accuracy_ml = checkpoint["minimum_loss_model_test_results"]["test_accuracy"]
            test_accuracy_ma = checkpoint["maximum_accuracy_model_test_results"]["test_accuracy"]
        
            cell_text = [
                [str(min_val_loss_epoch + 1), str(max_val_accuracy_epoch + 1)],
                [f"{val_loss_best_epoch_ml:.4f}", f"{val_loss_best_epoch_ma:.4f}"],
                [f"{val_accuracy_best_epoch_ml:.4f}", f"{val_accuracy_best_epoch_ma:.4f}"],
                [f"{test_loss_ml:.4f}", f"{test_loss_ma:.4f}"],
                [f"{test_accuracy_ml:.4f}", f"{test_accuracy_ma:.4f}"],
                [np.bincount(all_labels_ml, minlength=4), np.bincount(all_labels_ma, minlength=4)],
                [np.bincount(all_predictions_ml, minlength=4), np.bincount(all_predictions_ma, minlength=4),],
            ]
            
            row_labels = ["Best Epoch", "Val Loss", "Val Acc", "Test Loss", "Test Acc", "Test Labels", "Test Predictions"] if fold_idx == 0 else None
            
            mini_table = ax_table.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=[f"Min\nVal Loss", "Max\nVal Acc"],  
                loc='center',
                cellLoc='center',
                colWidths=[0.45, 0.45] #colWidths para travar o tamanho horizontal de todas as colunas
            )
            
            # Desativa o auto-ajuste de fonte automático que o Matplotlib faz quando o texto é muito grande
            mini_table.auto_set_font_size(False)
            mini_table.set_fontsize(9) # Tamanho fixo legível para os subplots de tamanho 22x10
            mini_table.scale(1.0, 1.4) 

            # Ajusta especificamente a altura das células do cabeçalho
            for col_idx in range(2):
                mini_table[0, col_idx].set_height(0.18)

            # Se não for o primeiro Fold, ajusta o alinhamento para compensar a falta de Row Labels
            if fold_idx > 0:
                # Remove espaços fantasmas que empurram tabelas sem rótulos laterais
                mini_table.scale(1.0, 1.0) 


        # ==========================================
        #  TABELA DE RESULTADO FINAL DE TESTE
        # ==========================================

        file_name = fold_files[-1]

        ax_table = axs[2, n_folds]
        ax_table.axis('off')
        cell_text = []

        # 1. Adicione o segundo valor dentro dos colchetes para cada Fold
        cell_text = [ 
            [f"{fold_accuracies_ml[fold_idx]:.4f}", f"{fold_accuracies_ma[fold_idx]:.4f}"] 
            for fold_idx in range(n_folds)
        ]

        fold_accuracies_mean_ml = f"{fold_accuracies_ml.mean():.4f}"
        fold_accuracies_std_ml = f"{fold_accuracies_ml.std():.4f}"

        # Suas novas métricas de média e desvio padrão para a nova coluna
        fold_accuracies_mean_ma = f"{fold_accuracies_ma.mean():.4f}"
        fold_accuracies_std_ma = f"{fold_accuracies_ma.std():.4f}"

        # 2. Adicione os valores correspondentes para as linhas Mean e Std
        cell_text.append([fold_accuracies_mean_ml, fold_accuracies_mean_ma])
        cell_text.append([fold_accuracies_std_ml, fold_accuracies_std_ma])

        row_labels = [f"Fold {fold_idx+1}" for fold_idx in range(n_folds)]
        row_labels.append("Mean")
        row_labels.append("Std")

        # 3. Adicione o nome da nova coluna na lista colLabels
        mini_table = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=[f"Min\nVal Loss", "Max\nVal Acc"],  
            loc='center',
            cellLoc='center',
            colWidths=[0.45, 0.45] 
        )

        # Desativa o auto-ajuste de fonte automático que o Matplotlib faz quando o texto é muito grande
        mini_table.auto_set_font_size(False)
        mini_table.set_fontsize(9) # Tamanho fixo legível para os subplots de tamanho 22x10
        mini_table.scale(1.0, 1.4) 

        num_rows = len(cell_text)

        # ---------------------------------------------------------
        # CORREÇÃO PARA OS CABEÇALHOS (colLabels) NÃO FICAREM ESPREMIDOS
        # ---------------------------------------------------------
        # Aumenta especificamente a altura das células da linha do cabeçalho (linha 0)
        for col_idx in range(2): # 2 é o número de colunas de dados
            mini_table[0, col_idx].set_height(0.15) # Ajuste este valor se precisar de mais espaço

        # Estilizando a Coluna 0 (Antiga) - Ajustado os índices de linha
        # O Matplotlib Table usa indexação baseada em (linha, coluna) considerando o cabeçalho
        mini_table[num_rows-1, 0].get_text().set_weight('bold')
        mini_table[num_rows, 0].get_text().set_weight('bold')
        mini_table[num_rows-1, 0].set_facecolor('#e6f2ff')
        mini_table[num_rows, 0].set_facecolor('#e6f2ff')

        # Estilizando a Coluna 1 (Nova Coluna)
        mini_table[num_rows-1, 1].get_text().set_weight('bold')
        mini_table[num_rows, 1].get_text().set_weight('bold')
        mini_table[num_rows-1, 1].set_facecolor('#e6f2ff')
        mini_table[num_rows, 1].set_facecolor('#e6f2ff')

        # Modificar os rótulos laterais (Row Labels na Coluna -1)
        mini_table[num_rows-1, -1].get_text().set_weight('bold')
        mini_table[num_rows, -1].get_text().set_weight('bold')


        # ==========================================
        #  TABELA DE NÚMERO DE AMOSTRAS POR FOLD
        # ==========================================

        ax_table = axs[1, n_folds]
        ax_table.axis('off')
        cell_text = []

        n_training_samples = np.array([])
        n_validation_samples = np.array([])
        n_test_samples = np.array([])
        n_total_samples = np.array([])

        for fold in range(1, n_folds + 1):

            fold_data = np.load(
                FOLDS_DIR / subject /  f"fold_{fold}.npz"
            )

            n_training_samples = np.append(n_training_samples, len(fold_data["train_indices"]))
            n_validation_samples = np.append(n_validation_samples, len(fold_data["val_indices"]))
            n_test_samples = np.append(n_test_samples, len(fold_data["test_indices"]))

        n_total_samples = np.sum([n_training_samples, n_validation_samples, n_test_samples], axis=0)

        cell_text = [ 
            [f"{n_training_samples[fold_idx]:.0f}", f"{n_validation_samples[fold_idx]:.0f}", f"{n_test_samples[fold_idx]:.0f}", f"{n_total_samples[fold_idx]:.0f}",  ] 
            for fold_idx in range(n_folds)
        ]

        # fold_accuracies_mean_ml = f"{fold_accuracies_ml.mean():.4f}"
        # fold_accuracies_std_ml = f"{fold_accuracies_ml.std():.4f}"

        # # Suas novas métricas de média e desvio padrão para a nova coluna
        # fold_accuracies_mean_ma = f"{fold_accuracies_ma.mean():.4f}"
        # fold_accuracies_std_ma = f"{fold_accuracies_ma.std():.4f}"

        # # 2. Adicione os valores correspondentes para as linhas Mean e Std
        # cell_text.append([fold_accuracies_mean_ml, fold_accuracies_mean_ma])
        # cell_text.append([fold_accuracies_std_ml, fold_accuracies_std_ma])

        row_labels = [f"Fold {fold_idx+1}" for fold_idx in range(n_folds)]
        # row_labels.append("Mean")
        # row_labels.append("Std")

        # 3. Adicione o nome da nova coluna na lista colLabels
        mini_table = ax_table.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=[f"Train", "Val", "Test", "Total"],  
            loc='center',
            cellLoc='center',
            colWidths=[0.24, 0.24, 0.24, 0.24] 
        )

            # Desativa o auto-ajuste de fonte automático que o Matplotlib faz quando o texto é muito grande
        mini_table.auto_set_font_size(False)
        mini_table.set_fontsize(9) # Tamanho fixo legível para os subplots de tamanho 22x10
        mini_table.scale(1.0, 1.4) 

        num_rows = len(cell_text)

        # ---------------------------------------------------------
        # CORREÇÃO PARA OS CABEÇALHOS (colLabels) NÃO FICAREM ESPREMIDOS
        # ---------------------------------------------------------
        # Aumenta especificamente a altura das células da linha do cabeçalho (linha 0)
        for col_idx in range(4): # 2 é o número de colunas de dados
            mini_table[0, col_idx].set_height(0.15) # Ajuste este valor se precisar de mais espaço

        # ==========================================
        # TABELA EXTRA
        # ==========================================
        
        ax_table = axs[0, n_folds]
        ax_table.axis('off')

        # ==========================================
        # SALVANDO OS DADOS
        # ===========================================
        # fig.subplots_adjust(top=0.94, bottom=0.05, left=0.08, right=0.95, hspace=0.4, wspace=0.3)
        plt.tight_layout()

        SAVE_FIG = OUTPUT_DIR / f"{branch}_branch" / f"{subject}"

        SAVE_FIG.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(SAVE_FIG / f"{branch}_{subject}.svg" , bbox_inches='tight')
        #plt.show()



if __name__ == "__main__":

    individual_subject_plots(branch="space1_space2_time_frequency" , subjects=["sub-01"])