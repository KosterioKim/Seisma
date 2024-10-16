import numpy as np
import matplotlib.pyplot as plt
import segyio
import matplotlib.patches as patches
from shapely.geometry import Polygon
import pandas as pd
import re


def read_seismic_cube(file_path):
    """
    :param file_path: Путь к файлу
    :return: возвращает куб в виде списков по Inlines, Xlines и Sampels (координаты x, y, z)
    а также словарю сейсмотрасс, где ключ это координата пересечения Inlines и Xlines, а значение это список значений сейсмотрасс
    """
    with segyio.open(file_path, 'r') as segyfile:
        x = list(segyfile.ilines)
        y = list(segyfile.xlines)
        z = list(segyfile.samples)
        cube = segyio.tools.cube(segyfile)
        traces = {f"{x[i]}_{y[j]}": cube[i][j][:] for i in range(len(x)) for j in range(len(y))}
    return x, y, z, traces

def pearson_correlation(list1, list2):
    """
    Функция рассчитывает коэффициент Пирсона (корреляцию) между двумя списками
    Если попадаются два нулевых списка, то возвращает 1
    """
    correlation = np.corrcoef(list1, list2)[0, 1]
    if np.isnan(correlation):
        return 1
    else:
        return correlation



def loc_min(values):
    """
    Функция ищет выбросы (локальные минимумы) с помощью скользящего интервала
    :param values: список линии корреции по Inlines или Xlines
    :return:
    """

    local_min = []
    for i in range(0, len(values) - 100 + 1, 10):
        # Выбираем текущий интервал
        current_interval = values[i:i + 100]
        # Находим минимум в текущем интервале
        min_value = min(current_interval)
        # Если минимум не равен 1, находим его индекс
        if min_value != 1:
            min_index = i + current_interval.index(min_value)
            # Проверяем, есть ли уже такой индекс в списке local_min
            if not any(min_index == existing_index for _, existing_index in local_min):
                local_min.append((min_value, min_index))
    return local_min[k][1], local_min[m][1]


def dild_polgon(ilines, xlines):
    """
    Построение полигонов по осям Х и У
    :param ilines: список координат Х
    :param xlines: список координат У
    :return: возвращает полигоны
    """
    coordinat_zona_ilines = []
    for i in range(len(ilines)):
        lin.append([])
        for j in range(len(xlines) - 1):
            lin[i].append(
                pearson_correlation(traces[f"{ilines[i]}_{xlines[j]}"], traces[f"{ilines[i]}_{xlines[j + 1]}"]))
        if all(element == 1 for element in lin[i]):
            coordinat_zona_ilines.append([f"{ilines[i]}_{xlines[0]}", f"{ilines[i]}_{xlines[-1]}"])
        else:
            min_index_1, min_index_2 = loc_min(lin[i])
            coordinat_zona_ilines.append([f"{ilines[i]}_{xlines[min_index_1]}", f"{ilines[i]}_{xlines[min_index_2]}"])
    poligon_ilines = []
    for i in range(len(coordinat_zona_ilines)):
        poligon_ilines.append(coordinat_zona_ilines[i][0])
    for l in coordinat_zona_ilines[::-1]:
        poligon_ilines.append(l[1])
    #print(poligon_ilines)

    lin_2 = []
    coordinat_zona_xlines = []
    for j in range(len(xlines)):
        lin_2.append([])
        for i in range(len(ilines) - 1):
            lin_2[j].append(
                pearson_correlation(traces[f"{ilines[i]}_{xlines[j]}"], traces[f"{ilines[i + 1]}_{xlines[j]}"]))
        if all(element == 1 for element in lin_2[j]):
            coordinat_zona_xlines.append([f"{ilines[0]}_{xlines[j]}", f"{ilines[-1]}_{xlines[j]}"])
        else:
            min_index_1, min_index_2 = loc_min(lin_2[j])
            coordinat_zona_xlines.append([f"{ilines[min_index_1]}_{xlines[j]}", f"{ilines[min_index_2]}_{xlines[j]}"])
    poligon_xlines = []
    for i in range(len(coordinat_zona_xlines)):
        poligon_xlines.append(coordinat_zona_xlines[i][0])
    for l in coordinat_zona_xlines[::-1]:
        poligon_xlines.append(l[1])

    return poligon_ilines, poligon_xlines

def expon(numbers):
    """
    функция сглаживает значения полигона с помощью экспоненциального сглаживания
    :param numbers: полигнон
    :return: сглаженный полигон
    """

    x_lines = []
    y_lines = []
    for i in range(len(numbers)):
        x_lines.append(round(int(numbers[i].split('_')[0]), 1))
        y_lines.append(round(int(numbers[i].split('_')[1]), 1))

    # Преобразование списка в объект Series
    data_x = pd.Series(x_lines)
    data_y = pd.Series(y_lines)

    # Применение экспоненциального сглаживания
    # alpha - это коэффициент сглаживания, который находится в диапазоне от 0 до 1
    alpha = 0.05
    x_i = data_x.ewm(alpha=alpha).mean()
    y_i = data_y.ewm(alpha=alpha).mean()

    p = []
    for i in range(len(x_i)):
        p.append(f"{round(x_i[i])}_{round(y_i[i])}")
    return p

def find_common_elements(list1, list2):
    # Преобразование списков в множества для поиска общих элементов
    set1 = set(list1)
    set2 = set(list2)

    # Нахождение пересечения множеств, что даст нам общие элементы
    common_elements = set1.intersection(set2)

    # Возврат списка общих элементов
    return list(common_elements)


def parse_coordinates(coord_str):
    x, y = map(float, re.findall(r'[-\d.]+', coord_str))
    return x, y

"""
Следующие 4 функции ищут угловые точки краевой зоны
"""
# 1 четверть
def find_min_coordinates(coords_list):
    # Преобразование списка строк в список кортежей координат
    coordinates = [parse_coordinates(coord) for coord in coords_list]

    # Нахождение точки с наименьшими координатами
    min_coord = min(coordinates, key=lambda x: (x[0], x[1]))

    return min_coord


# 2 четверть Функция для нахождения точки с наибольшей координатой по x и наименьшими координатами по y
def find_point_with_max_x_min_y(coords_list):
    # Преобразование списка строк в список кортежей координат
    coordinates = [parse_coordinates(coord) for coord in coords_list]

    # Нахождение точки с наибольшей координатой по x и наименьшими координатами по y
    max_x = max(coordinates, key=lambda x: x[0])
    min_y = min(coordinates, key=lambda x: x[1])

    # Возвращаем точку с наибольшей координатой по x из точек с наименьшей координатой по y
    result_point = min([point for point in coordinates if point[1] == min_y[1]], key=lambda x: x[0], default=None)

    return result_point


# 3 четверть Функция для нахождения точки с наибольшими координатами по x и y
def find_point_with_max_x_and_y(coords_list):
    # Преобразование списка строк в список кортежей координат
    coordinates = [parse_coordinates(coord) for coord in coords_list]

    # Нахождение точки с наибольшими координатами по x и y
    max_x = max(coordinates, key=lambda x: x[0])
    max_y = max(coordinates, key=lambda x: x[1])

    # Возвращаем точку с наибольшей координатой по x и y
    result_point = max(coordinates, key=lambda x: (x[0], x[1]))

    return result_point


# 4 четверть Функция для нахождения точки с наименьшей координатой по x и наибольшей по y
def find_point_with_min_x_and_max_y(coords_list):
    # Преобразование списка строк в список кортежей координат
    coordinates = [parse_coordinates(coord) for coord in coords_list]

    # Нахождение точки с наименьшей координатой по x
    min_x = min(coordinates, key=lambda x: x[0])

    # Нахождение точки с наибольшей координатой по y
    max_y = max(coordinates, key=lambda x: x[1])

    # Возвращаем точку с наименьшей координатой по x из точек с наибольшей координатой по y
    result_point = max([point for point in coordinates if point[0] == min_x[0]], key=lambda x: x[1], default=None)

    return result_point


def edge_zone(p_i, p_x, obshee):
    seredina_i = p_i[len(p_i) // 4].split('_')[0]
    seredina_x = p_x[len(p_x) // 4].split('_')[1]
    print(seredina_i, seredina_x)

    I = []
    II = []
    III = []
    IV = []
    for i in range(len(obshee)):
        if (int(obshee[i].split('_')[0]) > int(seredina_i)) and (int(obshee[i].split('_')[1]) > int(seredina_x)):
            I.append(obshee[i])
        elif (int(obshee[i].split('_')[0]) < int(seredina_i)) and (int(obshee[i].split('_')[1]) > int(seredina_x)):
            II.append(obshee[i])
        elif (int(obshee[i].split('_')[0]) < int(seredina_i)) and (int(obshee[i].split('_')[1]) < int(seredina_x)):
            III.append(obshee[i])
        elif (int(obshee[i].split('_')[0]) > int(seredina_i)) and (int(obshee[i].split('_')[1]) < int(seredina_x)):
            IV.append(obshee[i])
    krays = []
    #print('1 четверть ', I)
    krays.append(str(list(find_min_coordinates(I))[0])[:-2] + "_" + str(list(find_min_coordinates(I))[1])[:-2])

    #print('2 четверть ', II)
    krays.append(str(list(find_point_with_max_x_min_y(II))[0])[:-2] + "_" + str(list(find_point_with_max_x_min_y(II))[1])[:-2])

    #print('3 четверть ', III)
    krays.append(str(list(find_point_with_max_x_and_y(III))[0])[:-2] + "_" + str(list(find_point_with_max_x_and_y(III))[1])[:-2])

    #print('4 четверть ', IV)
    krays.append(str(list(find_point_with_min_x_and_max_y(IV))[0])[:-2] + "_" + str(
        list(find_point_with_min_x_and_max_y(IV))[1])[:-2])

    ppp_i = []
    ppp_x = []
    for i in range(4):
        ppp_i.append(p_i.index(krays[i]))
        ppp_x.append(p_x.index(krays[i]))
    granic_i = sorted(ppp_i)
    granic_x = sorted(ppp_x)
    poligon_finish = p_x[granic_x[0]:(granic_x[1] + 1)] + p_i[granic_i[2]:(granic_i[3] + 1)][::-1] + p_x[granic_x[2]:(
                granic_x[3] + 1)] + p_i[granic_i[0]: (granic_i[1] + 1)][::-1]

    return poligon_finish



def plot_polygon(coordinates_list, polygon_label):
    # Преобразование списка строк координат в список кортежей (x, y)
    polygon_points = [parse_coordinates(coord) for coord in coordinates_list]
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy
    #plt.plot(x, y, label=f'Полигон {polygon_label}')
    plt.fill(x, y, color='blue', label=f'Полигон {polygon_label}')
    plt.legend()




file_path = 'C:/HV/Seismic/1.1_edge_zone/Cube_TWT.segy'
ilines, xlines, samples, traces = read_seismic_cube(file_path)
poligon_ilines, poligon_xlines = dild_polgon(ilines, xlines, 1)
p_i = expon(poligon_ilines)
p_x = expon(poligon_xlines)

# Получение общих элементов
common_values = find_common_elements(p_i, p_x)
ez = edge_zone(p_i, p_x, common_values)

plot_polygon(ez, 'C')

