import random

def generate_csv(file_name, class_size, total_classes=3):
    print(f"Створення файлу: {file_name}")
    records = []

    sq1_size = 5
    sq2_size = 5
    offset = 10

    for _ in range(class_size):
        x = random.uniform(-sq1_size, sq1_size)
        y = random.uniform(-sq1_size, sq1_size)
        records.append(f"1,{x},{y}\n")

    for _ in range(class_size):
        x = random.uniform(sq1_size + offset, sq1_size + offset + 2 * sq2_size)
        y = random.uniform(-sq2_size, sq2_size)
        records.append(f"2,{x},{y}\n")

    count = 0
    while count < class_size:
        x = random.uniform(-30, 30)
        y = random.uniform(-30, 30)
        in_zone1 = (-sq1_size < x < sq1_size) and (-sq1_size < y < sq1_size)
        in_zone2 = (sq1_size + offset < x < sq1_size + offset + 2 * sq2_size) and (-sq2_size < y < sq2_size)
        if not (in_zone1 or in_zone2):
            records.append(f"0,{x},{y}\n")
            count += 1

    random.shuffle(records)
    with open(file_name, 'w') as f:
        f.writelines(records)

if __name__ == '__main__':
    generate_csv("saturn_data_train.csv", 1200)
    generate_csv("saturn_data_eval (1).csv", 120)
