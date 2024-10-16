import segyio
import numpy as np
import pandas as pd
import seaborn as sns


def rider():
    with segyio.open('C:/HV/Seismic/Geobody_cube.sgy', 'r') as segyfile:
        #print(segyfile)
        # Определяем размеры куба
        print(segyfile.xline.length, segyfile.iline.length, segyfile.samples.size)
        y = [i for i in segyfile.xlines]
        x = [i for i in segyfile.ilines]
        z = [i for i in segyfile.samples]
        print(x)
        print(y)
        print(z)
        cubic = segyio.tools.cube(segyfile)
        tras = {}
        for i in range(len(x)):
            for j in range(len(y)):
                tras[str(x[i]) + "_" + str(y[j])] = cubic[i][j][:]

    return x, y, z, tras

def show_srez(i):
    ai_cube = segyio.tools.cube('C:/HV/Seismic/Geobody_cube.sgy')  # загрузка куба
    # print(ai_cube)
    # Разрез или карта (срез)
    plt.figure(figsize=(7, 5), dpi=300)
    plt.imshow(ai_cube[i, :, :].transpose(), vmin=-0.5, vmax=0.5, cmap='seismic')
    # plt.xlabel('inline')
    # plt.ylabel('sample')
    plt.grid(ls=':', alpha=.5)
    plt.colorbar(shrink=0.4)
    plt.show()

def contains_negative(lst):
    return any(x == 0 for x in lst)

X, Y, Z, T = rider()
zero_index = []
for key in T:
    if contains_negative(T[key]):
        zero_index.append(key)

print(zero_index)

if len(zero_index) == 0:
    print('тест пройден успешно')
else:
    print('Значения скоростей должны быть положительными. Проверьте скоростную модель')

