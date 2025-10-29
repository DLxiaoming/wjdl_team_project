# wjdl_team_project
售前与交付团队代码库
## 成员标准工作流
### 1. 首次拉取仓库到本地
```
# 克隆仓库到本地（替换仓库地址为实际URL，在仓库主页点击“Code”获取）
git clone https://github.com/DLxiaoming/team_project.git

# 进入仓库目录
cd team_project
```

### 2. 创建自己的开发分支
```
# 确保本地主分支是最新的（首次克隆可省略，后续每次开发前建议执行）
git checkout main  # 切换到主分支
git pull origin main  # 拉取远程主分支的最新代码

# 创建并切换到自己的开发分支（分支名建议包含姓名和功能，如“zhangsan/login-feature”）
git checkout -b 你的分支名
```
### 3. 在自己的分支上开发代码
* 用编辑器修改代码（仅在自己的分支操作），确保只提交自己负责的内容。
* 定期提交代码到本地分支：
```
# 查看修改的文件
git status

# 添加修改到暂存区（.表示所有修改，也可指定具体文件）
git add .

# 提交到本地分支（备注清晰，说明改了什么）
git commit -m "完成XX功能：修复了XX问题/添加了XX逻辑"
```

### 4.推送自己的分支到远程 GitHub
```
# 首次推送该分支到远程（后续推送可直接用git push）
git push -u origin 你的分支名
```
核心命令
```
git pull
git add .
git commit
git push
```
## 开发完成后，申请合并到主分支（通过 Pull Request，PR）
1. 进入 GitHub 仓库主页 → 点击 “Pull requests” → “New pull request”。
2. 选择分支：
  左侧 “base”：选main（目标分支，即要合并到的主分支）。
  右侧 “compare”：选你自己的开发分支（如zhangsan/login-feature）。
3. 点击 “Create pull request” → 填写 PR 标题和描述（说明开发内容、测试情况等） → 再次点击 “Create pull request”。
## 管理员合并分支
1. 进入仓库的 “Pull requests” 页面，找到成员提交的 PR。
2. 检查代码：
  可直接在 GitHub 上查看代码修改（“Files changed” 标签）。
  若有问题，在 “Conversation” 标签留言，让成员修改后重新推送（成员推送后 PR 会自动更新）。
3. 确认无误后，点击 “Merge pull request” → “Confirm merge”，将分支代码合并到主分支。
4. 合并后可删除开发分支（点击 “Delete branch”，清理冗余分支）。

## 定期同步主分支代码：如果多人同时开发，你的分支可能落后于主分支，需定期同步避免冲突
```# 切换到自己的分支
git checkout 你的分支名

# 拉取主分支的最新代码到本地
git fetch origin main

# 合并主分支代码到自己的分支（解决冲突）
git merge origin/main
