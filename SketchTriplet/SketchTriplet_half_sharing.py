import torch.nn as nn

class BranchNet(nn.Module):
    def __init__(self, num_feat=100):
        super(BranchNet, self).__init__()
        self.num_feat = num_feat
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=15,
                stride=3,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.fc6 = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.fc7 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.feat = nn.Linear(512, num_feat)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.feat(x)
        return x

    def get_halfsharing(self, x):
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.feat(x)
        return x

    def get_branch(self, x):
        return self.forward(x)

class SketchTriplet(nn.Module):
    def __init__(self, branch_net):
        super(SketchTriplet, self).__init__()
        self.branch_net = branch_net

        self.conv1_a = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=15,
                stride=3,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv2_a = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2
            )
        )
        self.conv3_a = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )

    def ancSeq(self, x):
        x = self.conv1_a(x)
        x = self.conv2_a(x)
        x = self.conv3_a(x)
        # half sharing
        x = self.branch_net.get_halfsharing(x)
        return x

    def forward(self, anc_src, pos_src, neg_src):
        self.anc_src = anc_src  # anchor source     (sketch input)
        self.pos_src = pos_src  # positive source   (photograph edge input)
        self.neg_src = neg_src  # negative source   (photograph edge input)
        feat_a = self.ancSeq(self.anc_src)
        feat_p = self.branch_net(self.pos_src)
        feat_n = self.branch_net(self.neg_src)
        return feat_a, feat_p, feat_n

    def get_branch_sketch(self, x):
        return self.ancSeq(x)

    def get_branch_photo(self, x):
        return self.branch_net(x)

