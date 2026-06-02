# -*- coding: utf-8 -*-
"""
CodeFormer - restauracion local de imagenes en Windows.

Ejecutar desde el entorno virtual de Python 3.11:
    python main.py
"""

import glob
import os
import shutil
import site
import subprocess
import sys


CODEFORMER_FIDELITY = 0.7   # 0.0 = mas restauracion | 1.0 = mas fidelidad
BACKGROUND_ENHANCE = True   # True = mejora el fondo con Real-ESRGAN
FACE_UPSAMPLE = True        # True = upscale adicional en rostros

PINNED_PACKAGES = [
    'numpy==1.26.4',
    'opencv-python==4.8.1.78',
    'scipy==1.11.4',
    'scikit-image==0.21.0',
    'Pillow==10.4.0',
    'basicsr==1.4.2',
    'facexlib==0.3.0',
    'gfpgan==1.3.8',
    'realesrgan==0.3.0',
    'addict==2.4.0',
    'future==1.0.0',
    'lmdb==1.4.1',
    'pyyaml==6.0.2',
    'requests==2.28.1',
    'tqdm==4.67.1',
    'yapf==0.40.1',
    'lpips==0.1.4',
    'gdown==5.2.0',
    'tb-nightly',
    'einops==0.8.1',
]


def run(cmd):
    subprocess.run(cmd, check=True)


def clonar_repositorio():
    if os.path.exists('inference_codeformer.py'):
        print("Ya estamos en la carpeta correcta de CodeFormer.")
        return

    if not os.path.exists('CodeFormer'):
        print("Clonando repositorio CodeFormer...")
        run(['git', 'clone', 'https://github.com/sczhou/CodeFormer.git'])

    os.chdir('CodeFormer')
    if not os.path.exists('inference_codeformer.py') and os.path.exists('CodeFormer'):
        os.chdir('CodeFormer')

    print(f"Directorio de trabajo: {os.getcwd()}")


def asegurar_version_basicsr_local():
    """CodeFormer incluye un basicsr local. Sin version.py, el import falla."""
    version_txt = os.path.join('basicsr', 'VERSION')
    version_py = os.path.join('basicsr', 'version.py')

    if not os.path.exists(version_txt):
        print("  ! No se encontro basicsr/VERSION local")
        return

    with open(version_txt, encoding='utf-8') as f:
        version = f.read().strip()

    version_info = ', '.join(version.split('.'))
    content = (
        "# GENERATED VERSION FILE\n"
        f"__version__ = '{version}'\n"
        "__gitsha__ = 'local'\n"
        f"version_info = ({version_info})\n"
    )

    if not os.path.exists(version_py) or open(version_py, encoding='utf-8').read() != content:
        with open(version_py, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  OK basicsr/version.py local creado")
    else:
        print("  OK basicsr/version.py local ya existe")


def parchear_basicsr_instalado():
    """Corrige el import roto de basicsr 1.4.2 con torchvision moderno."""
    for sp in site.getsitepackages():
        degradations_path = os.path.join(sp, 'basicsr', 'data', 'degradations.py')
        if not os.path.exists(degradations_path):
            continue

        with open(degradations_path, encoding='utf-8') as f:
            content = f.read()

        fixed = content.replace(
            'from torchvision.transforms.functional_tensor import rgb_to_grayscale',
            'from torchvision.transforms.functional import rgb_to_grayscale',
        )

        if fixed != content:
            with open(degradations_path, 'w', encoding='utf-8') as f:
                f.write(fixed)
            print("  OK basicsr instalado parcheado")
        else:
            print("  OK basicsr instalado ya estaba parcheado")
        return

    print("  ! No se encontro degradations.py del basicsr instalado")


def instalar_dependencias():
    print("\nInstalando dependencias compatibles...")

    run([
        sys.executable, '-m', 'pip', 'install',
        'torch==2.1.0', 'torchvision==0.16.0',
        '--index-url', 'https://download.pytorch.org/whl/cu121',
        '--no-deps',
    ])

    constraints = os.path.abspath('constraints-local.txt')
    with open(constraints, 'w', encoding='utf-8') as f:
        f.write('\n'.join(PINNED_PACKAGES) + '\n')

    run([sys.executable, '-m', 'pip', 'install', '-c', constraints, *PINNED_PACKAGES])

    print("\nParcheando imports locales...")
    asegurar_version_basicsr_local()
    parchear_basicsr_instalado()


def descargar_modelos():
    modelo_cf = os.path.join('weights', 'CodeFormer', 'codeformer.pth')
    modelo_face = os.path.join('weights', 'facelib', 'detection_Resnet50_Final.pth')

    if os.path.exists(modelo_cf) and os.path.exists(modelo_face):
        print("\nModelos ya descargados, saltando...")
        return

    print("\nDescargando modelos de facelib...")
    run([sys.executable, 'scripts/download_pretrained_models.py', 'facelib'])

    print("\nDescargando modelo CodeFormer...")
    run([sys.executable, 'scripts/download_pretrained_models.py', 'CodeFormer'])


def preparar_imagenes():
    upload_folder = 'inputs/user_upload'
    os.makedirs(upload_folder, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("PASO: coloca tus imagenes en esta carpeta:")
    print(f"  {os.path.abspath(upload_folder)}")
    print("\nFormatos soportados: .jpg .jpeg .png .bmp .webp")
    print(f"{'=' * 60}")
    input("\nPresiona ENTER cuando hayas copiado las imagenes...")

    imagenes = [
        f for f in glob.glob(os.path.join(upload_folder, '*'))
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]

    if not imagenes:
        print("\nERROR: no se encontraron imagenes validas.")
        sys.exit(1)

    print(f"\nOK: {len(imagenes)} imagen(es) lista(s).")
    for img in imagenes:
        print(f"   - {os.path.basename(img)}")
    return upload_folder


def procesar_imagenes(upload_folder):
    print(f"\n{'=' * 60}")
    print("Procesando con CodeFormer...")
    print(f"  Fidelidad : {CODEFORMER_FIDELITY}")
    print(f"  Fondo     : {'mejorado con Real-ESRGAN' if BACKGROUND_ENHANCE else 'sin mejorar'}")
    print(f"  Upscale   : {'si' if FACE_UPSAMPLE else 'no'}")
    print(f"{'=' * 60}\n")

    cmd = [
        sys.executable, 'inference_codeformer.py',
        '-w', str(CODEFORMER_FIDELITY),
        '--input_path', upload_folder,
    ]
    if BACKGROUND_ENHANCE:
        cmd.extend(['--bg_upsampler', 'realesrgan'])
    if FACE_UPSAMPLE:
        cmd.append('--face_upsample')

    run(cmd)


def mostrar_resultados_y_guardar_zip():
    result_folder = f'results/user_upload_{CODEFORMER_FIDELITY}/final_results'
    ruta_resultados = os.path.abspath(result_folder)

    print(f"\n{'=' * 60}")
    print("RESULTADOS:")

    if os.path.exists(result_folder):
        for r in glob.glob(os.path.join(result_folder, '*')):
            print(f"  OK {os.path.basename(r)}")
        print(f"\nCarpeta: {ruta_resultados}")
    else:
        print("  Revisa la carpeta results manualmente.")

    zip_nombre = 'resultados_codeformer'
    zip_path = zip_nombre + '.zip'
    if os.path.exists(zip_path):
        os.remove(zip_path)

    try:
        shutil.make_archive(zip_nombre, 'zip', result_folder)
        print(f"\nZIP creado: {os.path.abspath(zip_path)}")
    except Exception as e:
        print(f"\nNo se pudo crear el ZIP: {e}")
        print(f"Tus imagenes estan en: {ruta_resultados}")

    print(f"{'=' * 60}")
    print("\nProceso completado.")


if __name__ == '__main__':
    print(f"Python detectado: {sys.version_info.major}.{sys.version_info.minor}")
    if sys.version_info[:2] != (3, 11):
        print("ERROR: usa Python 3.11. CodeFormer y sus dependencias no son estables en Python 3.14.")
        sys.exit(1)

    clonar_repositorio()
    instalar_dependencias()
    descargar_modelos()
    upload_folder = preparar_imagenes()
    procesar_imagenes(upload_folder)
    mostrar_resultados_y_guardar_zip()
