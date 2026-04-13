import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pandas as pd

#########################################################################################################################

class Model_Evaluator:

    def __init__(self, class_names):
        self.class_names = class_names

    def plot_learning_curves(self, history, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        epochs = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], label='Train Loss')
        plt.plot(epochs, history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], label='Train Acc')
        plt.plot(epochs, history['val_acc'], label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        save_path = f"{save_dir}/learning_curves.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Đã lưu Biểu đồ học tập tại: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        plt.tight_layout()
        save_path = f"{save_dir}/confusion_matrix.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Đã lưu Ma trận nhầm lẫn tại: {save_path}")

    def generate_quantitative_report(self, y_true, y_pred, save_dir):
        cm = confusion_matrix(y_true, y_pred)
        num_classes = len(self.class_names)
        
        sen_list, spe_list, f1_list = [], [], []

        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0    
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0    
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            
            sen_list.append(sensitivity * 100)
            spe_list.append(specificity * 100)
            f1_list.append(f1 * 100)

        overall_acc = np.trace(cm) / np.sum(cm) * 100

        report_str = "\n" + "="*75 + "\n"
        report_str += f"{'Quantitative evaluation of proposed method.':^75}\n"
        report_str += "-" * 75 + "\n"
        report_str += f"{'Class Label':^15} | {'Sen. (%)':^10} | {'Spe. (%)':^10} | {'F-score (%)':^12} | {'Acc. (%)':^10}\n"
        report_str += "-" * 75 + "\n"
        
        csv_data = [] 

        mid_idx = num_classes // 2 

        for i, cls in enumerate(self.class_names):
            csv_data.append({
                "Class Label": cls,
                "Sen. (%)": round(sen_list[i], 2),
                "Spe. (%)": round(spe_list[i], 2),
                "F-score (%)": round(f1_list[i], 2),
                "Acc. (%)": round(overall_acc, 2) if i == mid_idx else "" 
            })

            if i == mid_idx: 
                report_str += f"{cls:^15} | {sen_list[i]:^10.2f} | {spe_list[i]:^10.2f} | {f1_list[i]:^12.2f} | {overall_acc:^10.2f}\n"
            else:
                report_str += f"{cls:^15} | {sen_list[i]:^10.2f} | {spe_list[i]:^10.2f} | {f1_list[i]:^12.2f} | {'':^10}\n"
                
        report_str += "="*75 + "\n"
        print(report_str)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            txt_path = os.path.join(save_dir, "quantitative_report.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"Đã lưu báo cáo dạng Text tại: {txt_path}")

            csv_path = os.path.join(save_dir, "quantitative_report.csv")
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"Đã lưu báo cáo dạng CSV tại: {csv_path}")