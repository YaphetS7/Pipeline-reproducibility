import torch

from collections import OrderedDict

def compare_models(model_path1, model_path2):
    # Шаг 1: Загрузка моделей или state_dict'ов
    try:
        data1 = torch.load(model_path1, map_location=torch.device('cpu'))
        data2 = torch.load(model_path2, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Ошибка при загрузке файлов: {e}")
        return False

    # Шаг 2: Определение, что именно загружено (модель или state_dict)
    state_dict1 = data1 if isinstance(data1, OrderedDict) else data1.state_dict() if hasattr(data1, 'state_dict') else None
    state_dict2 = data2 if isinstance(data2, OrderedDict) else data2.state_dict() if hasattr(data2, 'state_dict') else None

    if state_dict1 is None or state_dict2 is None:
        print("Не удалось получить state_dict из загруженных данных")
        return False

    # Шаг 3: Проверка наличия одинаковых ключей
    if state_dict1.keys() != state_dict2.keys():
        print("Модели имеют разные ключи в state_dict")
        return False

    # Шаг 4: Сравнение значений для каждого ключа
    for key in state_dict1.keys():
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"Веса различаются для ключа: {key}")
            return False

    # Шаг 5: Если все проверки пройдены, модели идентичны
    print("Модели имеют идентичные веса")
    return True


# Шаг 7: Основная функция для запуска сравнения
def main():
    model_name = '1_mobnet_v2_4.pt'
    model_path1 = 'runs/1/' + model_name
    model_path2 = 'runs/2/' + model_name

    compare_models(model_path1, model_path2)

if __name__ == "__main__":
    main()