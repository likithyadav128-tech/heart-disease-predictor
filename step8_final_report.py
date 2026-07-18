# ============================================================
# Step 8: Final Report
# ============================================================
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, seaborn as sns, pickle
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

X_train=pd.read_csv("data/X_train.csv"); X_test=pd.read_csv("data/X_test.csv")
y_train=pd.read_csv("data/y_train.csv").squeeze(); y_test=pd.read_csv("data/y_test.csv").squeeze()
df=pd.read_csv("data/heart_cleaned.csv"); results=pd.read_csv("models/results.csv")

with open("models/logistic_regression.pkl","rb") as f: lr=pickle.load(f)
with open("models/random_forest.pkl","rb") as f:       rf=pickle.load(f)
with open("models/gradient_boosting.pkl","rb") as f:   gb=pickle.load(f)

print("=" * 60)
print("   HEART DISEASE RISK PREDICTION — FINAL REPORT")
print("=" * 60)
print(f"\n📋 Dataset    : {len(df)} patients, {df.shape[1]-1} features")
print(f"   Source      : fedesoriano — 5 UCI Datasets Combined")
print(f"   No Disease  : {(df['HeartDisease']==0).sum()}")
print(f"   Disease     : {(df['HeartDisease']==1).sum()}")
print(f"\n📊 Model Results:\n{results.to_string(index=False)}")
best=results.loc[results['ROC-AUC'].idxmax()]
print(f"\n🏆 Best Model : {best['Model']} (AUC={best['ROC-AUC']:.4f}, Acc={best['Accuracy']*100:.2f}%)")

# Final Dashboard
fig=plt.figure(figsize=(16,12))
fig.suptitle("Heart Disease Risk Prediction — Final Dashboard",fontsize=16,fontweight="bold",y=0.98)
gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35)

# Accuracy
ax1=fig.add_subplot(gs[0,0])
colors=["#e74c3c","#2ecc71","#3498db"]
bars=ax1.bar(results["Model"],results["Accuracy"]*100,color=colors,edgecolor="none",width=0.5)
for bar,v in zip(bars,results["Accuracy"]):
    ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.2,
             f"{v*100:.1f}%",ha="center",fontsize=9,fontweight="bold")
ax1.set_ylim(75,100); ax1.set_title("Accuracy",fontweight="bold")
ax1.tick_params(axis='x',rotation=15)

# ROC-AUC
ax2=fig.add_subplot(gs[0,1])
bars2=ax2.bar(results["Model"],results["ROC-AUC"],color=colors,edgecolor="none",width=0.5)
for bar,v in zip(bars2,results["ROC-AUC"]):
    ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.002,
             f"{v:.3f}",ha="center",fontsize=9,fontweight="bold")
ax2.set_ylim(0.85,1.0); ax2.set_title("ROC-AUC",fontweight="bold")
ax2.tick_params(axis='x',rotation=15)

# Metrics Heatmap
ax3=fig.add_subplot(gs[0,2])
m=results.set_index("Model")[["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]]
sns.heatmap(m,annot=True,fmt=".3f",cmap="YlGn",ax=ax3,linewidths=0.5,vmin=0.8,vmax=1.0)
ax3.set_title("All Metrics",fontweight="bold"); ax3.tick_params(axis='x',rotation=20)

# ROC Curves
ax4=fig.add_subplot(gs[1,0:2])
for model,name,color in[(lr,"Logistic Regression","#e74c3c"),(rf,"Random Forest","#2ecc71"),(gb,"Gradient Boosting","#3498db")]:
    prob=model.predict_proba(X_test)[:,1]; auc=roc_auc_score(y_test,prob)
    fpr,tpr,_=roc_curve(y_test,prob)
    ax4.plot(fpr,tpr,color=color,lw=2,label=f"{name} (AUC={auc:.3f})")
ax4.plot([0,1],[0,1],"k--",alpha=0.4)
ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR"); ax4.set_title("ROC Curves",fontweight="bold")
ax4.legend(loc="lower right")

# Feature Importance
ax5=fig.add_subplot(gs[1,2])
rf_imp=pd.Series(rf.feature_importances_,index=X_train.columns)
rf_imp.sort_values().tail(10).plot(kind="barh",color="#2ecc71",ax=ax5)
ax5.set_title("Top 10 Features (RF)",fontweight="bold"); ax5.set_xlabel("Importance")

plt.savefig("plots/14_final_dashboard.png",dpi=120,bbox_inches="tight")
plt.show()
print("\n✅ Saved: plots/14_final_dashboard.png")
print("\n" + "=" * 60)
print("✅ Step 8 Complete! Project fully done.")
print("=" * 60)