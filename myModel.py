import numpy as np
import os
class modelBuilder(object):
	def __init__(self):
		pass

	def get_wordnum_of_chapter(self, DocName):
		file_in = open(DocName, 'r', encoding='utf-8')

		text = ""
		for line in file_in:
			text += "".join(line.split('\n')) # 去除回车
		file_in.close()

		num = len(text.encode("UTF-8"))
		return num

	# 每个文档提取特征向量
	def build_feature_vector(self, DocName, label):

		function_word_list = ['之', '其', '或', '亦', '方', '于', '即', '皆', '因', '仍', 
							  '故', '尚', '呢', '了', '的', '着', '一', '不', '乃', '呀', 
							  '吗', '咧', '啊', '把', '让', '向', '往', '是', '在', '越', 
							  '再', '更', '比', '很', '偏', '别', '好', '可', '便', '就',
							  '但', '儿',                 # 42 个文言虚词
							  '又', '也', '都', '要',      # 高频副词
							  '这', '那', '你', '我', '他', # 高频代词
							  '来', '去', '道', '笑', '说' #高频动词
							  ] 
		# function_word_list = ["了", "的", "道", "也", "他", "是", "我", "你", "着", "又", "说", "来", "不"]
		feature_vector_list = []

		for function_word in function_word_list:
			
			find_flag = 0
			file_in = open(DocName, 'r', encoding='utf-8') #每次打开移动 cursor 到头部
			line = file_in.readline()
			while line:
				words = line[:-1].split('\t')
				if words[0] == function_word:
					total_words = self.get_wordnum_of_chapter(DocName)
					rate = float(words[1]) / total_words * 1000
					rate = float("%.6f" % rate)# 指定位数
					feature_vector_list.append(rate)
					# print words[0] + ' : ' + line

					file_in.close()
					find_flag = 1
					break
				line = file_in.readline()

			# 未找到词时向量为 0
			if not find_flag:
				feature_vector_list.append(0) 

		feature_vector_list.append(label)
		return feature_vector_list

	def make_Cao_trainset(self):
		Cao_trainset_list = []
		for loop in range(20, 30):
			feature = self.build_feature_vector(f'mywc/80_{loop}.txt', 1) #label 为 1 表示cao
			Cao_trainset_list.append(feature)
		# print Cao_trainset_list
		np.save('my_cao_trainset.npy', Cao_trainset_list)
		print('Cao trainset done')

	def make_Gao_trainset(self):
		Gao_trainset_list = []
		for loop in range(30, 40):
			feature = self.build_feature_vector(f'mywc/40_{loop}.txt', 2) #label 为 2 表示gao
			Gao_trainset_list.append(feature)
		# print Gao_trainset_list
		np.save('my_gao_trainset.npy', Gao_trainset_list)
		print('Gao trainset done')
	
	def make_Gui_trainset(self):
		Gui_trainset_list = []
		for loop in range(10, 20):
			feature = self.build_feature_vector(f'mywc/Guiyou_{loop}.txt', 3) #label 为 3 表示who
			Gui_trainset_list.append(feature)
		# print Gui_trainset_list
		np.save('my_gui_trainset.npy', Gui_trainset_list)
		print('Gui trainset done')

	def make_trainset(self):
		feature_cao = np.load('my_cao_trainset.npy')
		feature_gao = np.load('my_gao_trainset.npy')
		feature_gui = np.load('my_gui_trainset.npy')
		trainset = np.vstack((feature_cao, feature_gao, feature_gui))
		np.save('trainset.npy', trainset)
		print('Trainset done')

	def make_testset(self):
		testset_list = []
		file_name = os.listdir('mywc')

		for file in file_name:
			path = f'mywc/{file}'
			feature = self.build_feature_vector(path, 0) #无需 label，暂设为 0
			testset_list.append(feature)
		# print testset_list
		np.save('testset.npy', testset_list)
		print('Testset done')


if __name__ == '__main__':
	builder = modelBuilder()
	# print builder.build_feature_vector(1)

	builder.make_Cao_trainset() 	
	builder.make_Gao_trainset()
	builder.make_Gui_trainset()

	builder.make_trainset()
	builder.make_testset()
