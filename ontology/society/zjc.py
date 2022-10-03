# -*- coding: utf-8 -*-

from base import *

zjc = society.College('zjc', wholename='Zhijiang College, Zhejiang University of Technology', address='绍兴市柯桥区越州大道958号', taxno='12330000799606176D')
zjc.info = 'client no: 121102002900443710, phone:0575-81112517'

shuneng = society.Orgonization('shuneng', wholename='Shuneng Limited Company')
shuneng.owner = zhangxm = society.Person(wholename='Zhang Xiaoming')

with society:
    class ZJCTeacher(society.Teacher):
        empolied = [zjc]


    # yangm = Start('yangm', wholename='Yang Mi', birthday=datetime.date(1986, 9, 12), gender='female', info='teacher in science college')

    song = ZJCTeacher('songcw', wholename='Song Congwei', birthday=datetime.date(1986, 7, 5), gender='male', no='800052', level=7, salary=8000, position='assistant', info='teacher in science college')
    song.info += "6212261211004746600"

    jiangy = ZJCTeacher('jiangy', wholename='Jiang Yu', birthday=datetime.date(1986, 10, 5), gender='female', no='800026', level=10, salary=3000, position='lab assistant', info='lab assistant in science college')
    jiangy.comment = ['She is one of my most familiar friends.', 'She has two kids untill 2018.']

    chenjy = ZJCTeacher('chenjy', wholename='Chen Jingying', birthday=datetime.date(1986, 7, 5), gender='female', no='80005*', level=10, salary=3000, position='counselor', info='teacher in science college')
    chenjy.comment = ['Most beautiful lady in science college of zjc', 'She is good-hearted.']

    sunxq = ZJCTeacher('sunxq', wholename='Sun Xiaoquan', gender='male', position='secretary', info='teacher in science college')

    licn = ZJCTeacher('licn', wholename='Li Chunna', gender='female', position='assistant dean', info='teacher in science college')
    licn.comment = ['I don\'t know what is her attitude to me.', 'As a leader, she is sensitive to politics.']

    liubb = ZJCTeacher('liubb', wholename='Liu Binbin', gender='female', position='assistant', info='teacher in science college')

    peigh = ZJCTeacher('peigh', wholename='Pei Genhua', gender='male', position='assistant', info='teacher in science college, lived in Linping Town')

    huangyj = ZJCTeacher('huangyj', wholename='Huang Yujiao', gender='female', info='teacher in science college (from zjut), study ANN (theoretic)')
    
    liucy = ZJCTeacher('liucy', wholename='Liu Chunyan', gender='female', info='an excellent teacher')

    song.friend.extend([jiangy, chenjy])

    class love(ZJCTeacher >> ZJCTeacher):
        pass

    # class lovet(ZJCTeacher):
    #     equivalent_to = [ZJCTeacher & love.some(ZJCTeacher)]

    ZJCTeacher.is_a.append(love.some(ZJCTeacher))

sync_reasoner(debug=0)

print(list(chenjy.INDIRECT_friend))
print(chenjy.info)
print(love.some(ZJCTeacher) in ZJCTeacher.is_a)
print(ZJCTeacher in sunxq.INDIRECT_is_a)
print(love.some(ZJCTeacher) in sunxq.INDIRECT_is_instance_of)

