# ============================================================
# Step 6: Advanced Models — GB, Extra Trees, KNN & Weighted
#          Voting Ensemble (Best Accuracy, Compact Size)
# ============================================================
import pandas as pd, numpy as np, pickle, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)

X_train=pd.read_csv("data/X_train.csv"); X_test=pd.read_csv("data/X_test.csv")
y_train=pd.read_csv("data/y_train.csv").squeeze(); y_test=pd.read_csv("data/y_test.csv").squeeze()

def evaluate(name, model):
    yp=model.predict(X_test); yprob=model.predict_proba(X_test)[:,1]
    acc=accuracy_score(y_test,yp); prec=precision_score(y_test,yp)
    rec=recall_score(y_test,yp);   f1=f1_score(y_test,yp)
    auc=roc_auc_score(y_test,yprob)
    print(f"\n{'='*55}\n  {name}\n{'='*55}")
    print(f"  Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f} | Recall: {rec:.4f}")
    print(f"  F1 Score : {f1:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_test,yp,target_names=["No Disease","Disease"]))
    return yp,yprob,acc,prec,rec,f1,auc

# ── Random Forest (tuned) ──────────────────────────────────
print("🌲 Training Random Forest (tuned)...")
rf=RandomForestClassifier(n_estimators=100,max_depth=10,min_samples_leaf=4,
                           min_samples_split=2,random_state=42,n_jobs=-1)
rf.fit(X_train,y_train)
rf_pred,rf_prob,rf_acc,rf_prec,rf_rec,rf_f1,rf_auc=evaluate("RANDOM FOREST (tuned)",rf)
with open("models/random_forest.pkl","wb") as f: pickle.dump(rf,f)

cm=confusion_matrix(y_test,rf_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Greens",
            xticklabels=["No Disease","Disease"],yticklabels=["No Disease","Disease"])
plt.title("Random Forest — Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout(); plt.savefig("plots/08_rf_confusion_matrix.png"); plt.show()

# ── Gradient Boosting (tuned) ──────────────────────────────
print("\n⚡ Training Gradient Boosting (tuned)...")
gb=GradientBoostingClassifier(n_estimators=150,learning_rate=0.05,max_depth=3,
                               subsample=0.8,random_state=42)
gb.fit(X_train,y_train)
gb_pred,gb_prob,gb_acc,gb_prec,gb_rec,gb_f1,gb_auc=evaluate("GRADIENT BOOSTING (tuned)",gb)
with open("models/gradient_boosting.pkl","wb") as f: pickle.dump(gb,f)

cm=confusion_matrix(y_test,gb_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Oranges",
            xticklabels=["No Disease","Disease"],yticklabels=["No Disease","Disease"])
plt.title("Gradient Boosting — Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout(); plt.savefig("plots/09_gb_confusion_matrix.png"); plt.show()

# ── Extra Trees (compact — max_depth capped for smaller file) ─
print("\n🌳 Training Extra Trees (compact)...")
et=ExtraTreesClassifier(n_estimators=100,max_depth=8,random_state=42,n_jobs=-1)
et.fit(X_train,y_train)
et_pred,et_prob,et_acc,et_prec,et_rec,et_f1,et_auc=evaluate("EXTRA TREES (compact)",et)
with open("models/extra_trees.pkl","wb") as f: pickle.dump(et,f)

# ── K-Nearest Neighbors ─────────────────────────────────────
print("\n🔵 Training K-Nearest Neighbors...")
knn=KNeighborsClassifier(n_neighbors=15,weights='distance')
knn.fit(X_train,y_train)
knn_pred,knn_prob,knn_acc,knn_prec,knn_rec,knn_f1,knn_auc=evaluate("K-NEAREST NEIGHBORS",knn)
with open("models/knn.pkl","wb") as f: pickle.dump(knn,f)

# ── Weighted Voting Ensemble (Best Model: GB + ET + KNN×2) ──
print("\n🏆 Training Weighted Voting Ensemble (GB + ET + KNN, weights=[1,1,2])...")
voting = VotingClassifier(
    estimators=[('gb',gb),('et',et),('knn',knn)],
    voting='soft', weights=[1,1,2])
voting.fit(X_train,y_train)
vt_pred,vt_prob,vt_acc,vt_prec,vt_rec,vt_f1,vt_auc=evaluate("WEIGHTED VOTING ENSEMBLE",voting)
with open("models/voting_ensemble.pkl","wb") as f: pickle.dump(voting,f)

cm=confusion_matrix(y_test,vt_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm,annot=True,fmt="d",cmap="Purples",
            xticklabels=["No Disease","Disease"],yticklabels=["No Disease","Disease"])
plt.title("Weighted Voting Ensemble — Confusion Matrix")
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout(); plt.savefig("plots/09b_voting_confusion_matrix.png"); plt.show()

# ── Combined ROC Curve ──────────────────────────────────────
with open("models/logistic_regression.pkl","rb") as f: lr=pickle.load(f)
lr_prob=lr.predict_proba(X_test)[:,1]
lr_auc=roc_auc_score(y_test,lr_prob)
plt.figure(figsize=(7,6))
for prob,auc,name,color in[(lr_prob,lr_auc,"Logistic Regression","#e74c3c"),
                             (rf_prob,rf_auc,"Random Forest","#2ecc71"),
                             (gb_prob,gb_auc,"Gradient Boosting","#3498db"),
                             (vt_prob,vt_auc,"Voting Ensemble","#9b59b6")]:
    fpr,tpr,_=roc_curve(y_test,prob)
    plt.plot(fpr,tpr,color=color,lw=2,label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve — All Models")
plt.legend(); plt.tight_layout()
plt.savefig("plots/10_roc_comparison.png"); plt.show()

results=pd.read_csv("models/results.csv")
results = results[results["Model"]=="Logistic Regression"]  # keep only LR baseline
new_rows=pd.DataFrame({"Model":["Random Forest","Gradient Boosting","Extra Trees","KNN","Voting Ensemble"],
    "Accuracy":[round(rf_acc,4),round(gb_acc,4),round(et_acc,4),round(knn_acc,4),round(vt_acc,4)],
    "Precision":[round(rf_prec,4),round(gb_prec,4),round(et_prec,4),round(knn_prec,4),round(vt_prec,4)],
    "Recall":[round(rf_rec,4),round(gb_rec,4),round(et_rec,4),round(knn_rec,4),round(vt_rec,4)],
    "F1 Score":[round(rf_f1,4),round(gb_f1,4),round(et_f1,4),round(knn_f1,4),round(vt_f1,4)],
    "ROC-AUC":[round(rf_auc,4),round(gb_auc,4),round(et_auc,4),round(knn_auc,4),round(vt_auc,4)]})
pd.concat([results,new_rows],ignore_index=True).to_csv("models/results.csv",index=False)
print("\n📊 ALL MODELS:")
print(pd.read_csv("models/results.csv").to_string(index=False))
print("\n✅ Step 6 Complete!")