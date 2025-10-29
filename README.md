# wjdl_team_project
售前与交付团队代码库
# 克隆仓库到本地（替换仓库地址为实际URL，在仓库主页点击“Code”获取）
`git clone https://github.com/你的用户名/wjdl_team_project.git`

# 进入仓库目录
`cd team_project`

# 确保本地主分支是最新的（首次克隆可省略，后续每次开发前建议执行）
`git checkout main  # 切换到主分支`
`git pull origin main  # 拉取远程主分支的最新代码`

# 创建并切换到自己的开发分支（分支名建议包含姓名和功能，如“zhangsan/login-feature”）
`git checkout -b 你的分支名`

# 查看修改的文件
`git status`

# 添加修改到暂存区（.表示所有修改，也可指定具体文件）
`git add .`

# 提交到本地分支（备注清晰，说明改了什么）
`git commit -m "完成XX功能：修复了XX问题/添加了XX逻辑"`

# 首次推送该分支到远程（后续推送可直接用git push）
`git push -u origin 你的分支名`
