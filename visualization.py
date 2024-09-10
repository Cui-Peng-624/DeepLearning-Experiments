import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from IPython import display

class TrainingVisualizer:
    """在动画中绘制数据并记录最高的acc和最低的loss"""
    def __init__(self, xlabel=None, ylabel=None, title=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(7, 5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        self.legend = legend
        self.title = title  # 保存标题

        # 新增：记录最高acc和最低loss
        self.best_acc = float('-inf')  # 初始化为负无穷
        self.min_loss = float('inf')   # 初始化为正无穷

    def set_axes(self, ax, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """设置matplotlib的轴"""
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if legend:
            ax.legend(legend)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置x轴刻度为整数
        ax.grid()
        
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if self.X is None:
            self.X = [[] for _ in range(n)]
        if self.Y is None:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

                # 更新最高acc和最低loss
                if self.legend[i].lower().find('acc') != -1:  # 检查是否是acc数据
                    if b > self.best_acc:
                        self.best_acc = b
                if self.legend[i].lower().find('loss') != -1:  # 检查是否是loss数据
                    if b < self.min_loss:
                        self.min_loss = b

        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

        # 显示最高acc和最低loss
        if self.title:
            self.axes[0].set_title(self.title + f"\nBest Acc: {self.best_acc:.4f}, Min Loss: {self.min_loss:.4f}")
        else:
            self.axes[0].set_title(f"Best Acc: {self.best_acc:.4f}, Min Loss: {self.min_loss:.4f}")
        
        display.display(self.fig)
        display.clear_output(wait=True)

# 调用示例
# visualizer = TrainingVisualizer(xlabel='Epoch', ylabel='Value', title='Train TextRCNN', legend=['training_loss', "training_acc", "testing_loss", "testing_acc"]) # 初始化
# for epoch in range(num_epochs):
#     train_acc, train_loss = 
#     val_acc, val_loss = 
#     visualizer.add(epoch, [train_loss, train_acc, val_loss, val_acc])
