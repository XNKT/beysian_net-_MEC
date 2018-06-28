# -*- coding: utf-8 -*-
import random
import matplotlib
import matplotlib.pyplot as plt
import copy
u = [0, 0, 1, 1, 2, 3, 4, 5, 6, 7]
v = [1, 2, 3, 4, 5, 7, 6, 6, 7, 8]
fl = 500  # cpu频率
pl = 0.5  # cpu功率
pd = 0.01  # 空闲cpu功率
ps = 0.05  # 发送功率
rs = 2  # 发送速率
pr = 0.02  # 接收功率
rr = 2 # 接收速率
fc = 5000  # mecCPU频率


# 构造任务拓扑矩阵
def tasks():
    matrix = [([0] * 9) for i in range(9)]

    for i in range(len(u)):
        matrix[u[i]][v[i]] = random.uniform(0, 100)

    for j in range(9):
        matrix[j][j] = random.uniform(0, 3000)
    return matrix


# (前置)利用贝叶斯网络计算迁移策略(任务拓扑矩阵)
def beyesian(matrix):
    pro = [1]  # 各个子任务在本地执行的概率
    # I = [0]  # 各个子任务的执行位置
    conditional_pro = []

    for i in range(1, 9):
        if i== 5 or i==8:
            pro.append(1)
        else:
            e_qs = []
            e_pr = []
            wv = matrix[i][i]
            duv, pre_index = find_duv(matrix, i)
            el = pl * wv / fl
            ed = pd * wv / fc
            for j in range(len(duv)):
                e_qs.append(pd * duv[j] / rs + ps * duv[j] / rs)
                e_pr.append(pd * duv[j] / rr + pr * duv[j] / rr)
            table = find_conditional_probability_table(el, ed, e_qs, e_pr)
            # print table
            conditional_pro.append(table)
            if len(pre_index) == 1:
                pro.append(pro[pre_index[0]] * table[0][0] + (1 - pro[pre_index[0]]) * table[1][0])
                # pro.append(table[0][0] +   table[1][0])
            if len(pre_index) == 2:
                pro.append(
                    pro[pre_index[0]] * pro[pre_index[1]] * table[0][0] + (1 - pro[pre_index[0]]) * pro[pre_index[1]] *
                    table[1][0] +
                    pro[pre_index[0]] * (1 - pro[pre_index[1]]) * table[2][0] + (1 - pro[pre_index[0]]) * (
                    1 - pro[pre_index[1]]) * table[3][0])
                # pro.append(
                #      table[0][0] +
                #     table[1][0] +
                #      table[2][0] +  table[3][0])



    return pro


# 计算当前任务的条件概率表
def find_conditional_probability_table(el, ed, e_qs, e_pr):
    table = [([0] * 2) for i in range(len(e_qs) * 2)]
    if len(e_qs) == 1:
        table[0][0] = (e_qs[0] + ed) / (el + e_qs[0] + ed)
        table[0][1] = el / (el + e_qs[0] + ed)
        table[1][0] = ed / (el + +ed + e_pr[0])
        table[1][1] = (el + e_pr[0]) / (el + ed + e_pr[0])
    if len(e_qs) == 2:
        # u:0  z:0  v:0:1
        table[0][0] = (e_qs[0] + e_qs[1] + ed) / (el + e_qs[0] + e_qs[1] + ed)
        table[0][1] = el / (el + e_qs[0] + e_qs[1] + ed)
        # u:0  z:1  v:0:1
        table[1][0] = (ed + e_qs[0]) / (el + e_pr[1] + ed + e_qs[0])
        table[1][1] = (el + e_pr[1]) / (el + e_pr[1] + ed + e_qs[0])
        # u:1  z:0  v:0:1
        table[2][0] = (ed + e_qs[1]) / (el + e_pr[0] + ed + e_qs[1])
        table[2][1] = (el + e_pr[0]) / (el + e_pr[0] + ed + e_qs[1])
        # u:1  z:1  v:0:1
        table[3][0] = ed / (el + e_pr[0] + ed + e_pr[1])
        table[3][1] = (el + e_pr[0] + e_pr[1]) / (el + e_pr[0] + ed + e_pr[1])
    return table
#混合前后概率求I
def find_I(pro_pre,pro_back):
    I=[0]
    for i in range(1,9):
        if (pro_pre[i]+pro_back[8-i]) <((1-pro_pre[8-i])+(1-pro_back[8-i])):
            I.append(1)
        else:
            I.append(0)
    return I

#查找当前任务的前置任务的传输数据and前置任务的下标
def find_duv(matrix, i):
    duv = []
    pre_index = []
    for j in range(i):
        if matrix[j][i] > 0:
            duv.append(matrix[j][i])
            pre_index.append(j)
    return duv, pre_index


# (后置)利用贝叶斯网络计算迁移策略(任务拓扑矩阵)
def beyesian_back(matrix):
    pro = [1]  # 各个子任务在本地执行的概率
    # I = [0]  # 各个子任务的执行位置
    conditional_pro = []

    for i in range(8,-1,-1):
        if i== 5 or i==0:
            pro.append(1)
        else:
            e_qs = []
            e_pr = []
            wv = matrix[i][i]
            duv, pre_index = find_duv_back(matrix, i)
            el = pl * wv / fl
            ed = pd * wv / fc
            for j in range(len(duv)):
                e_qs.append(pd * duv[j] / rs + ps * duv[j] / rs)
                e_pr.append(pd * duv[j] / rr + pr * duv[j] / rr)
            table = find_conditional_probability_table(el, ed, e_pr, e_qs)
            # print table
            conditional_pro.append(table)
            if len(pre_index) == 1:
                pro.append(pro[8-pre_index[0]] * table[0][0] + (1 - pro[8-pre_index[0]]) * table[1][0])
                # pro.append(table[0][0] +   table[1][0])
            if len(pre_index) == 2:
                pro.append(
                    pro[pre_index[0]] * pro[pre_index[1]] * table[0][0] + (1 - pro[pre_index[0]]) * pro[pre_index[1]] *
                    table[1][0] +
                    pro[pre_index[0]] * (1 - pro[pre_index[1]]) * table[2][0] + (1 - pro[pre_index[0]]) * (
                    1 - pro[pre_index[1]]) * table[3][0])
                # pro.append(
                #      table[0][0] +
                #     table[1][0] +
                #      table[2][0] +  table[3][0])



    return pro


# 计算当前任务的条件概率表(后置)
def find_conditional_probability_table_back(el, ed, e_qs, e_pr):
    table = [([0] * 2) for i in range(len(e_qs) * 2)]
    if len(e_qs) == 1:
        table[0][0] = (e_qs[0] + ed) / (el + e_qs[0] + ed)
        table[0][1] = el / (el + e_qs[0] + ed)
        table[1][0] = ed / (el + +ed + e_pr[0])
        table[1][1] = (el + e_pr[0]) / (el + ed + e_pr[0])
    if len(e_qs) == 2:
        # u:0  z:0  v:0:1
        table[0][0] = (e_qs[0] + e_qs[1] + ed) / (el + e_qs[0] + e_qs[1] + ed)
        table[0][1] = el / (el + e_qs[0] + e_qs[1] + ed)
        # u:0  z:1  v:0:1
        table[1][0] = (ed + e_qs[0]) / (el + e_pr[1] + ed + e_qs[0])
        table[1][1] = (el + e_pr[1]) / (el + e_pr[1] + ed + e_qs[0])
        # u:1  z:0  v:0:1
        table[2][0] = (ed + e_qs[1]) / (el + e_pr[0] + ed + e_qs[1])
        table[2][1] = (el + e_pr[0]) / (el + e_pr[0] + ed + e_qs[1])
        # u:1  z:1  v:0:1
        table[3][0] = ed / (el + e_pr[0] + ed + e_pr[1])
        table[3][1] = (el + e_pr[0] + e_pr[1]) / (el + e_pr[0] + ed + e_pr[1])
    return table


# 查找当前任务的后置任务的传输数据and前置任务的下标
def find_duv_back(matrix, i):
    duv = []
    pre_index = []
    for j in range(i+1,9):
        if matrix[i][j] > 0:
            duv.append(matrix[i][j])
            pre_index.append(j)
    return duv, pre_index
# 计算总能耗
def total_energy(matrix, I):
    energy = matrix[0][0] * pl / fl
    for i in range(1, 9):
        e_qs = []
        e_pr = []
        wv = matrix[i][i]
        duv, pre_index = find_duv(matrix, i)
        el = pl * wv / fl
        ed = pd * wv / fc
        for j in range(len(duv)):
            e_qs.append(pd * duv[j] / rs + ps * duv[j] / rs)
            e_pr.append(pd * duv[j] / rr + pr * duv[j] / rr)

        E = (1 - I[i]) * el + I[i] * ed
        if len(pre_index) == 1:
            E += abs(I[pre_index[0]] - I[i]) * (e_qs[0] if I[pre_index[0]] > I[i]  else e_pr[0])
        if len(pre_index) == 2:
            falg = I[pre_index[0]] * 4 + I[pre_index[1]] * 2 + I[0]
            if falg == 1:
                E += (e_qs[0] + e_qs[1])
            elif falg == 2:
                E += e_pr[1]
            elif falg == 3:
                E += e_qs[0]
            elif falg == 4:
                E += e_pr[0]
            elif falg == 5:
                E += e_pr[1]
            elif falg == 6:
                E += (e_pr[0] + e_pr[1])
        energy += E
    return energy
#生成随机策略
def randomI():
    I=[0]
    for i in range(1,9):
        if i== 5 or i==8:
            I.append(0)
        else:
            I.append(random.randint(0,1))
    return I
#全部在本地执行的策略
def localI():
    I=[0,0,0,0,0,0,0,0,0]

    return I
#全部在mec服务器执行的策略
def serivceI():
    I=[0]
    for i in range(1,9):
        if i== 5 or i==8:
            I.append(0)
        else:
            I.append(1)
    return I

#贪婪策略
def greedy():
    I=[]
    for i in range(64):
        II=[]
        temp='{:06b}'.format(i)
        # print temp
        for j in range(6):
          ii=str(temp)[j:j+1]
          # print ii
          II.append(int(ii))
        I.append(II)
    return I
#贪婪策略加本地执行策略
def I_greedy_local(I):
    for i in range(64):
        I[i].insert(0,0)
        I[i].insert(5,0)
        I[i].insert(8,0)
    return I
#次贪婪策略
def trim(I_trim):
    trim_I=[I_trim]

    for i in range(len(I_trim)):
        I = copy.deepcopy(I_trim)
        if i!=0 and i!=5 and i!=8:
            I[i]=abs(1-I[i])
            trim_I.append(I)
    return trim_I
#贪婪策略的最优值
def greedy_best(task,I):
    index=0
    energy=total_energy(task, I[0])
    for i in range(1,len(I)):
        temp=total_energy(task, I[i])
        if temp<energy:
            energy=temp
            index=i
    return energy,I[index]
if __name__ == "__main__":
    print matplotlib.matplotlib_fname()
    num=1000
    I_energy=0
    RI_energy = 0
    LI_energy = 0
    SI_energy = 0
    greedy_I_energy=0
    local_weak_energy_I=0
    i_list = []
    I_list =[]
    RI_list = []
    LI_list = []
    SI_list = []
    best_list=[]
    I_greedy = greedy()
    I_greedys = I_greedy_local(I_greedy)
    for i in range(num):

        task = tasks()
        pro_pre= beyesian(task)
        pro_back=beyesian_back(task)
        print  (pro_pre,)
        print  (pro_back,)

        I_trim=find_I(pro_pre,pro_back)
        print I_trim
        I_energy_greedy,I=greedy_best(task,trim(I_trim))
        I_energy_greedy, I = greedy_best(task, trim(I))
        I_energy_greedy, I = greedy_best(task, trim(I))

        RI=randomI()
        LI=localI()
        SI=serivceI()
        # local_weak_energy,local_weak_I = greedy_best(task, trim(SI))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        # local_weak_energy, local_weak_I = greedy_best(task, trim(local_weak_I))
        greedy_energy, greedy_I = greedy_best(task, I_greedys)
        print (I,)
        print (greedy_I,)
        print('\n')
        # local_weak_energy_I+=local_weak_energy
        I_energy+=I_energy_greedy
        RI_energy += total_energy(task, RI)
        LI_energy += total_energy(task, LI)
        SI_energy += total_energy(task, SI)
        greedy_I_energy+=greedy_energy
        if i%(num/10)==0:
            # print i
            i_list.append(i)
            I_list.append(I_energy)
            RI_list.append(RI_energy)
            LI_list.append(LI_energy)
            SI_list.append(SI_energy)
            best_list.append(greedy_I_energy)

    print I_energy
    print RI_energy
    print LI_energy
    print SI_energy
    print greedy_I_energy
    print greedy_I_energy-I_energy
    # print local_weak_energy_I

    i_array=[i_list[0],i_list[1],i_list[2],i_list[3],i_list[4],i_list[5],i_list[6],i_list[7],i_list[8],i_list[9]]
    I_array=[I_list[0],I_list[1],I_list[2],I_list[3],I_list[4],I_list[5],I_list[6],I_list[7],I_list[8],I_list[9]]
    RI_array = [RI_list[0],RI_list[1],RI_list[2],RI_list[3],RI_list[4],RI_list[5],RI_list[6],RI_list[7],RI_list[8],RI_list[9]]
    LI_array = [LI_list[0],LI_list[1],LI_list[2],LI_list[3],LI_list[4],LI_list[5],LI_list[6],LI_list[7],LI_list[8],LI_list[9]]
    SI_array = [SI_list[0],SI_list[1],SI_list[2],SI_list[3],SI_list[4],SI_list[5],SI_list[6],SI_list[7],SI_list[8],SI_list[9]]
    best_array=[best_list[0],best_list[1],best_list[2],best_list[3],best_list[4],best_list[5],best_list[6],best_list[7],best_list[8],best_list[9]]
    plt.figure(figsize=(8, 8))
    plt.plot(i_array, I_array, marker='o',color='green', label=u'贝叶斯迁移策略')
    plt.plot(i_array, RI_array, marker='*',color='red', label=u'随机迁移策略')
    plt.plot(i_array, LI_array, marker='.',color='coral', label=u'全部本地迁移策略')
    plt.plot(i_array, SI_array,marker='x', color='blue', label=u'全部服务器迁移策略')

    plt.plot(i_array, best_array, marker='>',color='black', label=u'贪婪策略')
    plt.legend(loc=0)
    plt.xlabel('tasks')
    plt.ylabel('energy')
    plt.show()