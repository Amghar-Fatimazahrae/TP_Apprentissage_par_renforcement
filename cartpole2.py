import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# 1. Créer l'environnement
env = gym.make("CartPole-v1", render_mode="human")

# 2. Définir le modèle RL
model = PPO("MlpPolicy", env, verbose=1)

# 3. Entraîner l'agent
total_timesteps = 10000

print("Entraînement en cours...")
model.learn(total_timesteps=total_timesteps)
print("Entraînement terminé !")

# 4. Tester l'agent sur 5 épisodes
num_episodes = 5
rewards_list = []

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        total_reward += reward

    rewards_list.append(total_reward)

    print(f"Épisode {ep+1} - Récompense : {total_reward}")

env.close()

# 5. Visualiser les récompenses
plt.figure(figsize=(8, 4))
plt.plot(range(1, num_episodes + 1), rewards_list, marker='o')
plt.title("Récompense totale par épisode")
plt.xlabel("Épisode")
plt.ylabel("Récompense")
plt.grid(True)
plt.show()
