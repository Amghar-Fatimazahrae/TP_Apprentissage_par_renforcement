
import gymnasium as gym
from stable_baselines3 import PPO

# 1. Créer l'environnement
env = gym.make("CartPole-v1")

# 2. Définir le modèle RL
model = PPO("MlpPolicy", env, verbose=1)

# 3. Entraîner l'agent
print("Entraînement en cours...")
model.learn(total_timesteps=10000)
print("Entraînement terminé !")

# 4. Créer un environnement pour visualiser le test
test_env = gym.make("CartPole-v1", render_mode="human")

# 5. Tester l'agent entraîné
obs, info = test_env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = test_env.step(action)

    done = terminated or truncated
    total_reward += reward

# 6. Afficher le résultat
print(f"Récompense totale : {total_reward}")

test_env.close()
env.close()