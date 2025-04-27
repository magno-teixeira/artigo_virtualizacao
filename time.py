import os
import time
import software

os.system("cls|clear")

inicio = time.time()
print("Início = ", inicio)

software.programa()

fim = time.time()
print("Fim = ", fim)
print("Tempo de execução do programa (Fim - Início) = ", fim - inicio)

normalizado = (fim - inicio)*100000

print(f"Tempo de execução normalizado (x100000) = {normalizado:.1f} segundos")
