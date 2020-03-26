import torch.nn as nn

class SketchTriplet(nn.Module):
    def __init__(self, num_feat=100):
        super(SketchTriplet, self).__init__()
        self.num_feat = num_feat
        #-------------------------------------
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
        self.conv4_a = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )
        self.conv5_a = nn.Sequential(
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
        self.fc6_a = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.fc7_a = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.feat_a = nn.Linear(512, num_feat)
        #-------------------------------------
        self.conv1_p = nn.Sequential(
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
        self.conv2_p = nn.Sequential(
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
        self.conv3_p = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )
        self.conv4_p = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )
        self.conv5_p = nn.Sequential(
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
        self.fc6_p = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.fc7_p = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.feat_p = nn.Linear(512, num_feat)
        # -------------------------------------
        self.conv1_n = nn.Sequential(
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
        self.conv2_n = nn.Sequential(
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
        self.conv3_n = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )
        self.conv4_n = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU()
        )
        self.conv5_n = nn.Sequential(
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
        self.fc6_n = nn.Sequential(
            nn.Linear(256 * 5 * 5, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.fc7_n = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.55)
        )
        self.feat_n = nn.Linear(512, num_feat)

    def ancSeq(self, x):
        x = self.conv1_a(x)
        x = self.conv2_a(x)
        x = self.conv3_a(x)
        x = self.conv4_a(x)
        x = self.conv5_a(x)
        x = x.view(x.size(0), -1)
        x = self.fc6_a(x)
        x = self.fc7_a(x)
        x = self.feat_a(x)
        return x

    def posSeq(self, x):
        x = self.conv1_p(x)
        x = self.conv2_p(x)
        x = self.conv3_p(x)
        x = self.conv4_p(x)
        x = self.conv5_p(x)
        x = x.view(x.size(0), -1)
        x = self.fc6_p(x)
        x = self.fc7_p(x)
        x = self.feat_p(x)
        return x

    def negSeq(self, x):
        x = self.conv1_n(x)
        x = self.conv2_n(x)
        x = self.conv3_n(x)
        x = self.conv4_n(x)
        x = self.conv5_n(x)
        x = x.view(x.size(0), -1)
        x = self.fc6_n(x)
        x = self.fc7_n(x)
        x = self.feat_n(x)
        return x

    def forward(self, anc_src, pos_src, neg_src):
        self.anc_src = anc_src  # anchor source
        self.pos_src = pos_src  # positive source
        self.neg_src = neg_src  # negative source
        feat_a = self.ancSeq(self.anc_src)
        feat_p = self.posSeq(self.pos_src)
        feat_n = self.negSeq(self.neg_src)
        return feat_a, feat_p, feat_n

    def get_branch_a(self, x):
        return self.ancSeq(x)

    def get_branch_p(self, x):
        return self.posSeq(x)

    def get_branch_n(self, x):
        return self.negSeq(x)