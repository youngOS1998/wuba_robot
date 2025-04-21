import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class SimpleCPG:
    def __init__(self, omega1=1, omega2=1.05, alpha=0.5, beta=0.2, gamma=0.1):
        """
        初始化CPG模型参数
        omega1, omega2: 振荡器的自然频率
        alpha, beta: 内部振荡器的动力学参数
        gamma: 振荡器之间的耦合强度
        """
        self.omega1 = omega1
        self.omega2 = omega2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.x1 = 0  # 振荡器1的状态
        self.x2 = 0  # 振荡器2的状态
        self.time = 0

    def update(self, dt):
        """
        更新CPG状态
        dt: 时间步长
        """
        dx1 = self.omega1 * self.x2 - self.alpha * self.x1 - self.gamma * (self.x1 - self.x2)
        dx2 = self.omega2 * self.x1 - self.alpha * self.x2 - self.gamma * (self.x2 - self.x1)
        
        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        self.time += dt
        
        return self.x1, self.x2

    def simulate(self, total_time=10, dt=0.01):
        """
        模拟CPG一段时间
        total_time: 总模拟时间
        dt: 时间步长
        """
        num_steps = int(total_time / dt)
        positions = np.zeros((num_steps, 2))
        
        for i in range(num_steps):
            positions[i, :] = self.update(dt)
            
        return positions, np.linspace(0, total_time, num_steps)

    def simulate_and_plot(self, total_time=10, dt=0.01):
        """
        模拟CPG并绘制结果
        total_time: 总模拟时间
        dt: 时间步长
        """
        positions, _ = self.simulate(total_time, dt)
        plt.figure(figsize=(10, 5))
        plt.plot(positions[:, 0], label='Oscillator 1', linewidth=2)
        plt.plot(positions[:, 1], label='Oscillator 2', linewidth=2, linestyle='--')
        plt.title('Phase Space Plot of a Simple Central Pattern Generator (CPG)')
        plt.xlabel('Time Steps')
        plt.ylabel('Phase')
        plt.legend()
        plt.grid(True)
        plt.show()

# 使用模型并绘制结果
cpg_model = SimpleCPG()
cpg_model.simulate_and_plot()