# 发布到 GitHub：本机操作步骤

远程仓库已命名为 [Yuxiang5555/AI6103-group-project](https://github.com/Yuxiang5555/AI6103-group-project)。本地文件夹仍可叫 `unimc-model-optimization`，不需要因为远程名称变化而移动或重命名目录。建议课程项目协作阶段先使用 Private 仓库，邀请组员，再由团队决定何时公开和采用什么许可证。

当前目录已整理源码、历史实验说明、英文首页、三种子结果、Gate、LoRA、诊断、图表、依赖和 CPU CI。23 个原始根目录 Python 文件保持原始字节，原研究目录没有修改。数据、基座权重、缓存和个人交接包不在普通 Git 文件中。

## 1. 创建 GitHub 空仓库

远程空仓库已经创建；无需再次新建。若以后从头操作，可在 [GitHub 新建仓库页面](https://github.com/new) 选择名称 `AI6103-group-project`，不要初始化 README、`.gitignore` 或 LICENSE，因为本地已有对应内容或许可状态说明。

本项目的 HTTPS 地址为 `https://github.com/Yuxiang5555/AI6103-group-project.git`。

## 2. 在本机 Windows PowerShell 初始化

当前工作区已经完成本节的初始化和首次提交，无需重复。若使用ZIP解压到一个新目录，再执行本节。

下面必须在这份独立仓库目录运行，不要在包含其他课程实验的工作区根目录运行 `git add .`：

```powershell
cd "C:\path\to\unimc-model-optimization"
git init -b main
git config core.autocrlf false
git add .
git status --short
git diff --cached --stat
git commit -m "Add UniMC optimization methods, experiments, and diagnostics"
```

如果 Git 首次使用提示身份未配置，只在本仓库设置自己的名字和邮箱，然后再次执行 commit：

```powershell
git config user.name "Your Name"
git config user.email "YOUR_GITHUB_EMAIL"
```

邮箱可用自己 GitHub 账号提供的 noreply 地址；不要填写仓库整理者或虚构身份。

## 3. 连接并推送

如果本地已有指向旧名称的 origin，更新地址：

```powershell
git remote set-url origin https://github.com/Yuxiang5555/AI6103-group-project.git
git remote -v
git push -u origin main
```

如果是在没有 origin 的全新本地副本中，改用 `git remote add origin https://github.com/Yuxiang5555/AI6103-group-project.git`。HTTPS 推送需要 GitHub 认证，按 Git Credential Manager/浏览器提示登录自己的账号即可；不要在脚本里写密码或 token。当前本机没有 GitHub CLI，所以本指南不依赖 `gh`。

## 4. 页面检查与协作

打开仓库，确认英文首页、结果表和图能正常显示。在 Actions 中查看首次 CPU contract tests；本地通过不等于 GitHub 托管工作流已经运行。邀请组员加入协作，分支和 Pull Request 用于后续改动。

后续更新在同一目录运行：

```powershell
git status
git add README.md docs results reports
git diff --cached --stat
git commit -m "Update experiment documentation and results"
git push
```

修改代码时按实际路径添加相应文件。上传前查看 staged diff，避免误把其他课程作业或运行产物混入仓库。

## 5. 大文件与交接包

普通仓库保留源码、表格和图。基座约826 MB，应通过 Hugging Face 下载；缓存和数据通过准备脚本重建。已有训练头、adapter 或交接包如需分发，可由团队另行放到适当的附件/Release 存储，记录版本与SHA，不要直接取消全部忽略规则。

GitHub 普通 Git 文件超过100 MiB会被阻止，网页上传单文件限制25 MiB。参考 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。本模板的文件远小于这些限制。

完整流程依据 [GitHub 官方本地项目上传指南](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=windows)。当前工作区已有首次提交和仓库名称更新提交；origin 已改为新仓库地址。本次整理未执行推送。
