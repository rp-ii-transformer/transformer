import subprocess, sys

def run(script_path):
    print(f"🛠️  Executando {script_path} ...")
    p = subprocess.run([sys.executable, script_path])
    if p.returncode != 0:
        print(f"❌ Erro em {script_path} (code={p.returncode})")
        sys.exit(p.returncode)

if __name__ == "__main__":
    run("scripts/train.py")
    run("scripts/run_inference.py")
    print("🎉 Todos os scripts rodaram com sucesso!")
