# Phase 3 Enhanced: Production-Ready Pollock AI
# suzuさん用 - 海外展開対応版

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import random
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from datetime import datetime
import json

class AdvancedPollockSimulator:
    """
    高度なポロック風ドリッピングシミュレーター
    Phase 2の改良 + 新機能追加
    """
    
    def __init__(self, canvas_size=(512, 512)):  # 高解像度対応
        self.canvas_size = canvas_size
        self.canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
        self.layer_history = []  # レイヤー履歴保存
        
    def create_advanced_drip_path(self, start_x, start_y, paint_type="acrylic", brush_size="medium"):
        """
        絵具タイプと筆のサイズを考慮したドリッピングパス生成
        
        Parameters:
        - paint_type: "acrylic", "oil", "watercolor"
        - brush_size: "small", "medium", "large"
        """
        # 絵具タイプ別のパラメータ
        paint_params = {
            "acrylic": {"viscosity": 0.85, "gravity": 0.25, "spread": 1.0},
            "oil": {"viscosity": 0.92, "gravity": 0.18, "spread": 0.8},
            "watercolor": {"viscosity": 0.65, "gravity": 0.35, "spread": 1.5}
        }
        
        # 筆サイズ別のパラメータ
        brush_params = {
            "small": {"thickness_range": (0.5, 2.0), "steps_range": (100, 200)},
            "medium": {"thickness_range": (1.0, 4.0), "steps_range": (150, 300)},
            "large": {"thickness_range": (2.0, 8.0), "steps_range": (200, 400)}
        }
        
        paint_config = paint_params[paint_type]
        brush_config = brush_params[brush_size]
        
        path = []
        current_x = float(start_x)
        current_y = float(start_y)
        
        # 初期パラメータ設定
        velocity_x = random.uniform(-3.0, 3.0)
        velocity_y = random.uniform(-2.0, 2.0)
        thickness = random.uniform(*brush_config["thickness_range"])
        max_steps = random.randint(*brush_config["steps_range"])
        
        # 風の影響（新機能）
        wind_x = random.uniform(-0.1, 0.1)
        wind_y = random.uniform(-0.05, 0.05)
        
        for step in range(max_steps):
            # 物理シミュレーション
            velocity_y += paint_config["gravity"] * 0.08
            velocity_x *= paint_config["viscosity"]
            velocity_y *= paint_config["viscosity"]
            
            # 風の影響追加
            velocity_x += wind_x
            velocity_y += wind_y
            
            # 位置更新
            current_x += velocity_x
            current_y += velocity_y
            
            # 境界処理（改良版）
            if current_x < 0:
                current_x = 0
                velocity_x = abs(velocity_x) * 0.6
            elif current_x >= self.canvas_size[1]:
                current_x = self.canvas_size[1] - 1
                velocity_x = -abs(velocity_x) * 0.6
                
            if current_y < 0:
                current_y = 0
                velocity_y = abs(velocity_y) * 0.6
            elif current_y >= self.canvas_size[0]:
                current_y = self.canvas_size[0] - 1
                # 下端に到達したら横に流れる
                velocity_x += random.uniform(-1.0, 1.0)
                velocity_y *= 0.3
            
            # 絵具の厚さ変化
            thickness *= 0.997
            if thickness < 0.2:
                break
            
            # より複雑な揺らぎ
            noise_factor = paint_config["spread"]
            fluctuation_x = random.uniform(-0.8, 0.8) * noise_factor
            fluctuation_y = random.uniform(-0.4, 0.4) * noise_factor
            current_x += fluctuation_x
            current_y += fluctuation_y
            
            # 筆の動きの変化
            if random.random() < 0.08:
                velocity_x += random.uniform(-1.5, 1.5)
                velocity_y += random.uniform(-1.0, 1.0)
            
            path.append((int(current_x), int(current_y), thickness))
            
        return path
    
    def create_textured_splash(self, center_x, center_y, color, paint_type="acrylic"):
        """
        絵具タイプに応じたテクスチャード飛散効果
        """
        splash_configs = {
            "acrylic": {"density": 15, "size_range": (1, 4), "distance_range": (5, 20)},
            "oil": {"density": 8, "size_range": (2, 6), "distance_range": (3, 15)},
            "watercolor": {"density": 25, "size_range": (1, 3), "distance_range": (8, 30)}
        }
        
        config = splash_configs[paint_type]
        num_splashes = random.randint(config["density"] // 2, config["density"])
        
        for _ in range(num_splashes):
            distance = random.uniform(*config["distance_range"])
            angle = random.uniform(0, 2 * math.pi)
            
            splash_x = int(center_x + distance * math.cos(angle))
            splash_y = int(center_y + distance * math.sin(angle))
            
            if (0 <= splash_x < self.canvas_size[1] and 
                0 <= splash_y < self.canvas_size[0]):
                
                splash_size = random.randint(*config["size_range"])
                
                # 透明度を考慮した色の重ね合わせ
                alpha = random.uniform(0.3, 0.8)
                current_color = self.canvas[splash_y, splash_x]
                new_color = [int(c * alpha + current_color[i] * (1 - alpha)) 
                           for i, c in enumerate(color)]
                
                cv2.circle(self.canvas, (splash_x, splash_y), splash_size, new_color, -1)
    
    def draw_advanced_drip(self, path, color, paint_type="acrylic"):
        """
        高度なドリッピング描画（テクスチャとブレンディング対応）
        """
        if len(path) < 2:
            return
        
        for i in range(len(path) - 1):
            x1, y1, thickness1 = path[i]
            x2, y2, thickness2 = path[i + 1]
            
            avg_thickness = int((thickness1 + thickness2) / 2)
            
            # グラデーション効果
            if paint_type == "watercolor":
                # 水彩風の薄い線
                alpha = 0.6
                overlay = self.canvas.copy()
                cv2.line(overlay, (x1, y1), (x2, y2), color, max(1, avg_thickness))
                self.canvas = cv2.addWeighted(self.canvas, 1 - alpha, overlay, alpha, 0)
            else:
                # 通常の描画
                cv2.line(self.canvas, (x1, y1), (x2, y2), color, max(1, avg_thickness))
            
            # 飛散効果
            if random.random() < 0.4:
                self.create_textured_splash(x1, y1, color, paint_type)
    
    def create_professional_artwork(self, style="classic", complexity="medium"):
        """
        プロフェッショナル品質のアートワーク生成
        
        Parameters:
        - style: "classic", "modern", "experimental"
        - complexity: "simple", "medium", "complex"
        """
        self.canvas = np.zeros((self.canvas_size[0], self.canvas_size[1], 3), dtype=np.uint8)
        
        style_configs = {
            "classic": {
                "layers": 3,
                "drips_per_layer": 8,
                "colors": [(200, 50, 50), (50, 150, 200), (255, 200, 50), (100, 200, 100)],
                "paint_types": ["oil", "acrylic"]
            },
            "modern": {
                "layers": 4,
                "drips_per_layer": 12,
                "colors": [(255, 100, 150), (100, 255, 200), (150, 100, 255), (255, 255, 100)],
                "paint_types": ["acrylic", "watercolor"]
            },
            "experimental": {
                "layers": 5,
                "drips_per_layer": 15,
                "colors": [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                          for _ in range(8)],
                "paint_types": ["acrylic", "oil", "watercolor"]
            }
        }
        
        complexity_modifiers = {
            "simple": 0.7,
            "medium": 1.0,
            "complex": 1.4
        }
        
        config = style_configs[style]
        modifier = complexity_modifiers[complexity]
        
        num_layers = int(config["layers"] * modifier)
        drips_per_layer = int(config["drips_per_layer"] * modifier)
        
        for layer in range(num_layers):
            print(f"Creating {style} style layer {layer + 1}/{num_layers}...")
            
            paint_type = random.choice(config["paint_types"])
            brush_size = random.choice(["small", "medium", "large"])
            
            for _ in range(drips_per_layer):
                # より戦略的な開始位置
                if layer == 0:
                    # 最初のレイヤーは上部中央から
                    start_x = random.randint(self.canvas_size[1] // 4, 3 * self.canvas_size[1] // 4)
                    start_y = random.randint(0, self.canvas_size[0] // 3)
                else:
                    # 後のレイヤーはより広範囲から
                    start_x = random.randint(0, self.canvas_size[1] - 1)
                    start_y = random.randint(0, self.canvas_size[0] // 2)
                
                color = random.choice(config["colors"])
                path = self.create_advanced_drip_path(start_x, start_y, paint_type, brush_size)
                self.draw_advanced_drip(path, color, paint_type)
            
            # レイヤー間処理
            if layer < num_layers - 1:
                self.apply_layer_effects(paint_type)
        
        return self.canvas
    
    def apply_layer_effects(self, paint_type):
        """レイヤー間の効果適用"""
        if paint_type == "watercolor":
            # 水彩風のにじみ効果
            self.canvas = cv2.GaussianBlur(self.canvas, (3, 3), 0)
        elif paint_type == "oil":
            # 油絵風のテクスチャ
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.canvas = cv2.filter2D(self.canvas, -1, kernel)
            self.canvas = np.clip(self.canvas, 0, 255).astype(np.uint8)
    
    def save_artwork(self, filename=None, metadata=None):
        """作品保存（メタデータ付き）"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pollock_artwork_{timestamp}.png"
        
        # 画像保存
        cv2.imwrite(filename, self.canvas)
        
        # メタデータ保存
        if metadata:
            meta_filename = filename.replace('.png', '_metadata.json')
            with open(meta_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"Artwork saved: {filename}")
        return filename

class ProductionPollockGAN:
    """
    本格運用向けポロックGAN
    """
    
    def __init__(self, image_size=512, latent_dim=128):
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.simulator = AdvancedPollockSimulator(canvas_size=(image_size, image_size))
        
        # より高性能なモデル構築
        self.generator = self.build_production_generator()
        self.discriminator = self.build_production_discriminator()
        
        # 最適化されたオプティマイザー
        self.gen_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.999)
        
        # 訓練履歴
        self.training_history = {"gen_loss": [], "disc_loss": []}
    
    def build_production_generator(self):
        """本格運用向けジェネレーター"""
        model = keras.Sequential([
            # 入力層（拡張）
            layers.Dense(8 * 8 * 1024, input_shape=(self.latent_dim,)),
            layers.Reshape((8, 8, 1024)),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # Progressive upsampling with skip connections
            layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            
            # 出力層（高品質化）
            layers.Conv2D(3, (7, 7), padding='same', activation='tanh'),
        ], name="production_generator")
        
        return model
    
    def build_production_discriminator(self):
        """本格運用向けディスクリミネーター"""
        model = keras.Sequential([
            # 入力層
            layers.Conv2D(16, (4, 4), strides=(2, 2), padding='same', 
                         input_shape=(self.image_size, self.image_size, 3)),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # Progressive downsampling
            layers.Conv2D(32, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            
            # 出力層
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid'),
        ], name="production_discriminator")
        
        return model
    
    def generate_diverse_training_data(self, batch_size=16):
        """多様な学習データ生成"""
        images = []
        styles = ["classic", "modern", "experimental"]
        complexities = ["simple", "medium", "complex"]
        
        for _ in range(batch_size):
            style = random.choice(styles)
            complexity = random.choice(complexities)
            
            self.simulator.canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            artwork = self.simulator.create_professional_artwork(style, complexity)
            
            # データ拡張
            if random.random() < 0.3:
                # 回転
                angle = random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((self.image_size // 2, self.image_size // 2), angle, 1)
                artwork = cv2.warpAffine(artwork, M, (self.image_size, self.image_size))
            
            if random.random() < 0.2:
                # 明度調整
                hsv = cv2.cvtColor(artwork, cv2.COLOR_BGR2HSV)
                hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], random.uniform(0.8, 1.2))
                artwork = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            image = artwork.astype(np.float32) / 255.0
            image = (image - 0.5) * 2  # [-1, 1] に正規化
            images.append(image)
        
        return np.array(images)
    
    def train_production(self, epochs=1000, batch_size=16, save_interval=100):
        """本格訓練"""
        print("=== Production Pollock GAN Training ===")
        print(f"Image size: {self.image_size}x{self.image_size}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        for epoch in range(epochs):
            # 学習データ生成
            real_images = self.generate_diverse_training_data(batch_size)
            
            # 訓練ステップ
            gen_loss, disc_loss = self.train_step(real_images)
            
            # 履歴記録
            self.training_history["gen_loss"].append(float(gen_loss))
            self.training_history["disc_loss"].append(float(disc_loss))
            
            # 進捗表示
            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Generator Loss: {gen_loss:.4f}")
                print(f"  Discriminator Loss: {disc_loss:.4f}")
            
            # サンプル生成・保存
            if epoch % save_interval == 0 and epoch > 0:
                self.save_checkpoint(epoch)
                self.generate_sample_gallery(epoch)
    
    @tf.function
    def train_step(self, real_images):
        """最適化された訓練ステップ"""
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss
    
    def generator_loss(self, fake_output):
        """改良されたジェネレーター損失"""
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_output, labels=tf.ones_like(fake_output)))
    
    def discriminator_loss(self, real_output, fake_output):
        """改良されたディスクリミネーター損失"""
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=real_output, labels=tf.ones_like(real_output)))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=fake_output, labels=tf.zeros_like(fake_output)))
        return real_loss + fake_loss
    
    def generate_sample_gallery(self, epoch, num_samples=9):
        """サンプルギャラリー生成"""
        noise = tf.random.normal([num_samples, self.latent_dim])
        generated_images = self.generator(noise, training=False)
        generated_images = (generated_images + 1) / 2
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        for i in range(num_samples):
            axes[i].imshow(generated_images[i])
            axes[i].set_title(f'Generated #{i+1}', fontsize=12)
            axes[i].axis('off')
        
        plt.suptitle(f'Epoch {epoch} - AI Pollock Gallery', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'pollock_gallery_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_checkpoint(self, epoch):
        """チェックポイント保存"""
        checkpoint_dir = f"pollock_checkpoints/epoch_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.generator.save_weights(f"{checkpoint_dir}/generator.h5")
        self.discriminator.save_weights(f"{checkpoint_dir}/discriminator.h5")
        
        # 訓練履歴保存
        with open(f"{checkpoint_dir}/training_history.json", 'w') as f:
            json.dump(self.training_history, f)
        
        print(f"Checkpoint saved: {checkpoint_dir}")
    
    def create_portfolio_collection(self, num_pieces=16, save_individual=True):
        """ポートフォリオ用作品集生成"""
        print("Creating portfolio collection...")
        
        noise = tf.random.normal([num_pieces, self.latent_dim])
        generated_images = self.generator(noise, training=False)
        generated_images = (generated_images + 1) / 2
        
        # 個別保存
        if save_individual:
            portfolio_dir = "pollock_portfolio"
            os.makedirs(portfolio_dir, exist_ok=True)
            
            for i in range(num_pieces):
                image = (generated_images[i].numpy() * 255).astype(np.uint8)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{portfolio_dir}/pollock_masterpiece_{i+1:02d}.png", image_bgr)
        
        # 4x4グリッド表示
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        for i in range(num_pieces):
            axes[i].imshow(generated_images[i])
            axes[i].set_title(f'Masterpiece #{i+1}', fontsize=14)
            axes[i].axis('off')
        
        plt.suptitle('AI Pollock Portfolio Collection', fontsize=24, y=0.98)
        plt.tight_layout()
        plt.savefig('pollock_portfolio_collection.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return generated_images

def demo_advanced_features():
    """新機能デモンストレーション"""
    print("=== Advanced Pollock AI Demo ===")
    
    # 高解像度シミュレーター
    simulator = AdvancedPollockSimulator(canvas_size=(512, 512))
    
    # 各スタイルのデモ
    styles = ["classic", "modern", "experimental"]
    
    for style in styles:
        print(f"\nCreating {style} style artwork...")
        artwork = simulator.create_professional_artwork(style=style, complexity="medium")
        
        # 作品保存
        metadata = {
            "style": style,
            "complexity": "medium",
            "canvas_size": simulator.canvas_size,
            "timestamp": datetime.now().isoformat()
        }
        filename = simulator.save_artwork(f"demo_{style}_style.png", metadata)
        
        # 表示
        plt.figure(figsize=(12, 12))
        rgb_canvas = cv2.cvtColor(artwork, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_canvas)
        plt.title(f'{style.title()} Style Pollock Art (512x512)', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main_production():
    """本格運用メイン関数"""
    print("=== Phase 3 Enhanced: Production-Ready Pollock AI ===")
    
    # デモ実行
    demo_advanced_features()
    
    # 本格GAN訓練（軽量版）
    print("\n=== Starting Production GAN Training ===")
    pollock_gan = ProductionPollockGAN(image_size=256, latent_dim=128)  # 実行時間考慮
    
    print("Model Architecture:")
    print(f"Generator parameters: {pollock_gan.generator.count_params():,}")
    print(f"Discriminator parameters: {pollock_gan.discriminator.count_params():,}")
    
    # 軽量訓練（デモ用）
    pollock_gan.train_production(epochs=100, batch_size=8, save_interval=50)
    
    # 最終作品集生成
    print("\n=== Creating Final Portfolio ===")
    portfolio = pollock_gan.create_portfolio_collection(num_pieces=9)
    
    print("\n=== Phase 3 Enhanced Complete! ===")
    print("Ready for Phase 4: Web UI Development")

if __name__ == "__main__":
    main_production()