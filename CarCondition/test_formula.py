# Тест новой формулы расчета итогового процента

## Старая формула: (Чистота + Целостность) / 2
## Новая формула: Чистота×30% + Целостность×70%

def old_formula(clean, integrity):
    return (clean + integrity) / 2

def new_formula(clean, integrity):
    return (clean * 0.3) + (integrity * 0.7)

# Примеры сравнения
test_cases = [
    (90, 50),  # Чистая, но поврежденная
    (50, 90),  # Грязная, но целая
    (80, 80),  # Средне и то и то
    (100, 100), # Идеальная
    (30, 30),   # Плохая
]

print("| Чистота | Целостность | Старая формула | Новая формула | Разница |")
print("|---------|-------------|----------------|---------------|---------|")

for clean, integrity in test_cases:
    old = old_formula(clean, integrity)
    new = new_formula(clean, integrity)
    diff = new - old
    print(f"| {clean:7}% | {integrity:11}% | {old:14.1f}% | {new:13.1f}% | {diff:+6.1f}% |")

print("\n✅ Новая формула делает целостность важнее чистоты!")
print("💡 Теперь поврежденная машина получит более низкий общий балл")