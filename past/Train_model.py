import torch
import torch.utils.data.dataloader as dl
import torch.optim as optim
import os
from time import gmtime, strftime

torch.multiprocessing.set_sharing_strategy('file_system')

from DSNet import DSNet

class Config:
    """
    对训练模型进行配置
    """
    def __init__(self):
        torch.manual_seed(0)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.train_loader = None
        self.model = None
        self.optimizer = None
        self.training_procedure = None
        self.experiment_name = "test_the_model"

    def train(self, epoch):
        print("初始化的时候执行train")
        self.training_procedure(self.model, self, self.train_loader, self.optimizer, epoch)


    def initialize_experiment(self):
        if not os.path.exists("./experiments"):
            os.mkdir("./experiments")
        self.experiment_path = os.path.join("./experiments", str(strftime("%Y-&%m-%d %H-%M-%S", gmtime())+" "+self.experiment_name))
        os.mkdir(self.experiment_path)
        self.checkpoint_path = os.path.join(self.experiment_path, "checkpoints")
        os.mkdir(self.checkpoint_path)

        # for path in self.paths_to_copy: # 调用了子类中的参 将那几个跟训练有关的文件夹进行保存  # TODO 这里是一些文件夹的操作 可暂时不看 需要的时候再看
        #     destination = os.path.join(self.experiment_path,os.path.basename(path)) # 文件的名字，注意不是 文件夹的名字哦 这里是将对应的文件进行保存
        #     if os.path.isdir(path): # 如果是文件夹 就用copytree拷贝文件夹
        #         shutil.copytree(path, destination)
        #     else:                   # 如果是文件 就用copyfile拷贝文件
        #         shutil.copyfile(path, destination)


class configModel(Config):
    def __init__(self, root_path = "/"):
        from past.dealDataSet import Generate_data
        from past.TrainUtils import train_network
        super(configModel, self).__init__()
        self.batch_size = 1
        self.lr = 0.005
        self.shuffle = True
        self.epochs = 20
        self.log_interval = 10

        self.model = DSNet().to(self.device)
        self.train_loader = dl.DataLoader(dataset=Generate_data(), batch_size=self.batch_size,
                                          shuffle=self.shuffle,
                                          num_workers=1, drop_last=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")
        self.training_procedure = train_network

        self.initialize_experiment()


def main():
    cfg = configModel()
    for epoch in range(cfg.epochs):
        cfg.train(epoch)

if __name__ == '__main__':
    main()



