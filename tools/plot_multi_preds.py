import matplotlib.pyplot as plt

# 横坐标：每条 trajectory 上进行预测的次数
x = [1, 110]

# 纵坐标：对应 accuracy
y = [0.5344696969696969, 0.5416666666666666]

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='o', linestyle='-', linewidth=2, color='orange')

# 添加文字注释（调低位置）
for i in range(len(x)):
    plt.text(x[i], y[i] + 0.0001, f"{y[i]:.4f}", ha='center', va='top', fontsize=10)

# 设置标签
plt.xlabel("Number of predictions per trajectory")
plt.ylabel("Accuracy")
plt.title("Effect of Multiple Predictions per Trajectory (Window=30)")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig("effect_of_multiple_predictions.png", dpi=300, bbox_inches='tight')