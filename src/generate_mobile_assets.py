"""
Gerador de ícones e splash screen oficiais do App Mobile CardioIA Fase 4
Grupo: Data Health Solutions (FIAP)
"""

import os
from PIL import Image, ImageDraw

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOBILE_ASSETS = os.path.join(ROOT_DIR, "mobile", "assets")


def generate_mobile_branding():
    os.makedirs(MOBILE_ASSETS, exist_ok=True)

    # 1. Ícone 512x512 no tema FIAP (#030305 + #ED145B)
    icon = Image.new("RGB", (512, 512), color=(3, 3, 5))
    draw = ImageDraw.Draw(icon)
    draw.rounded_rectangle([32, 32, 480, 480], radius=90, outline=(237, 20, 91), width=14)
    draw.ellipse([136, 136, 376, 376], fill=(237, 20, 91))
    draw.ellipse([190, 190, 322, 322], fill=(15, 23, 42))
    icon.save(os.path.join(MOBILE_ASSETS, "icon.png"))

    # 2. Splash Screen 800x1200
    splash = Image.new("RGB", (800, 1200), color=(3, 3, 5))
    sdraw = ImageDraw.Draw(splash)
    sdraw.rounded_rectangle([200, 400, 600, 800], radius=64, outline=(237, 20, 91), width=10)
    sdraw.ellipse([300, 500, 500, 700], fill=(237, 20, 91))
    splash.save(os.path.join(MOBILE_ASSETS, "splash.png"))


if __name__ == "__main__":
    generate_mobile_branding()
