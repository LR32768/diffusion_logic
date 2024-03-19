import numpy as np
import matplotlib.pyplot as plt
import math

def cal_exact_score(data, alpha, x):
    diff = data * np.sqrt(alpha) - x
    d2 = np.sum(diff**2, axis=1) / (2*(1-alpha))
    d2 -= d2.min()
    kernel = np.exp(- d2)
    kernel /= np.sum(kernel)
    # print(f"dist square: {d2}")
    # print(f"kernel: {kernel}")

    # use kernel as weights to average data
    weight_x0 = data * kernel[:, np.newaxis]
    weight_x0 = np.sum(weight_x0, axis=0)
    # print(x,":",score[0])

    score = x / np.sqrt(1-alpha) - np.sqrt(alpha / (1-alpha)) * weight_x0
    print(-score)
    return -score, weight_x0

def cal_exact_density(data, alpha, x):
    diff = data * np.sqrt(alpha) - x
    d2 = np.sum(diff**2, axis=1)
    kernel = np.exp(- d2 / (2*(1-alpha)))
    density = np.sum(kernel)
    return density

def score_vis(data):
    # Create a toy dataset and visualize the flow lines

    # tune alpha from 0 to 0.99, make a gif to show the change of flow lines
    for alpha in np.linspace(0, 0.99, 20):
        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)

        U = np.zeros(X.shape)
        V = np.zeros(X.shape)
        P = np.zeros(X.shape)
        # Draw the data points
        plt.figure(figsize=(6, 6))
        plt.scatter(data[:, 1] * np.sqrt(alpha), data[:, 0]* np.sqrt(alpha), c='r', s=100)

        # Use the exact score function to calculate the flow lines, then visualize them
        for i in range(len(x)):
            for j in range(len(y)):
                score, _ = cal_exact_score(data, alpha, np.array([x[i], y[j]]))
                U[i, j] = score[1]
                V[i, j] = score[0]
                P[i, j] = cal_exact_density(data, alpha, np.array([x[i], y[j]]))


        plt.title("alpha = {:.2f}".format(alpha))
        plt.contourf(X, Y, P, alpha=0.3, levels=20, cmap="RdBu_r")
        plt.streamplot(X, Y, U, V, density=2, color='b')
        plt.show()
        plt.close()

def parity_score_vis(N=4, x=None):
    # Generate parity vectors
    data = []
    for i in range(2**(N-1)):
        z = np.zeros(N)
        for j in range(N):
            z[j] = (i >> j) & 1
        z[-1] = np.sum(z[:N]) % 2
        data.append(z*2.0 - 1)
    data = np.array(data)

    if x is None:
        x = np.random.randn(N) / 5

    # tune alpha from 0 to 0.99, make a gif to show the change of flow lines
    for alpha in np.linspace(0, 0.99, 20):
        score = cal_exact_score(data, alpha, x)
        idx = np.arange(N)

        # set y range to be [-1.1, 1.1]
        plt.ylim(-1.1, 1.1)
        plt.plot(idx, np.zeros(N), color='k', linestyle='--')
        plt.plot(idx, score, 'o-', color='r')
        plt.plot(idx, x, 'o-', color='b')
        plt.title("alpha = {:.2f}".format(alpha))
        plt.show()
        plt.close()


# Write a sample simulation to start from gaussian distribution, and then use the exact score function to update the samples
def sample_from_exact_score(data, num_steps=100, dim=8, schedule="linear", noise_sample=False):

    def linear_beta_schedule(timesteps):
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        #return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        return np.linspace(beta_start, beta_end, timesteps)

    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        #x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        x = np.linspace(0, timesteps, steps)
        #alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # return torch.clip(betas, 0, 0.999)
        return np.clip(betas, 0, 0.999)

    x = np.random.randn(dim)
    if schedule == "linear":
        beta = linear_beta_schedule(num_steps)
    elif schedule == "cosine":
        beta = cosine_beta_schedule(num_steps)
    alpha = 1 - beta
    alpha_cumprod = np.cumprod(alpha)
    alpha_cumprod_prev = np.concatenate([[1], alpha_cumprod[:-1]])
    # print(beta)
    # print(alpha)
    # print(alpha_cumprod)
    # print(alpha_cumprod_prev)

    # Start from x, each time update the diffusion process with the exact score function
    # Reverse the process to get the samples
    for i in reversed(range(1, num_steps)):
        score, weight_x0 = cal_exact_score(data, alpha_cumprod[i], x)
        # new_mean = (x - (1-alpha[i])/np.sqrt(1-alpha_cumprod[i])*score) / np.sqrt(alpha[i])
        # x = new_mean
        x = (1-alpha_cumprod_prev[i]) / (1- alpha_cumprod[i]) * np.sqrt(alpha[i]) * x - \
            beta[i] * np.sqrt(alpha[i]) / (1- alpha_cumprod[i]) * weight_x0
        if noise_sample:
            z = np.random.randn(dim)
            noise_scale = np.sqrt(beta[i] * (1-alpha_cumprod_prev[i])/(1-alpha_cumprod[i]))
            x += noise_scale * z
        # Do clip
        x = np.clip(x, -1.2, 1.2)
        print(f"step {i}, beta {beta[i]}: {x}")

        if i % 10 == 0:
            plt.ylim(-2, 2)
            plt.plot(x, 'o-', color='b')
            plt.plot(score, '-', color='r')
            plt.plot(np.zeros(dim), '--', color='g')
            plt.title(f"step {i}")
            plt.show()
            plt.close()
    return x


if __name__ == "__main__":
    # data = np.array([[-1,-1], [1,1], [1,-1]])
    # score_vis(data)

    N = 8
    data = []
    # for i in range(2 ** (N - 1)):
    #     z = np.zeros(N)
    #     for j in range(N):
    #         z[j] = (i >> j) & 1
    #     z[-1] = np.sum(z[:N]) % 2
    #     data.append(z * 2.0 - 1)
    # data = np.array(data)

    for i in range(2 ** (N)):
        z = np.zeros(N)
        for j in range(N):
            z[j] = (i >> j) & 1
        if z.sum() == N // 2:
            data.append(z * 2.0 - 1)
    data = np.array(data)

    print(data)
    sample_from_exact_score(data, num_steps=1000, dim=N,
                            schedule="linear", noise_sample=True)