# Bash 操作命令完全手册

智慧水利项目常用命令速查手册

---

## 目录

1. [Python 环境管理](#python-环境管理)
2. [Git 版本控制](#git-版本控制)
3. [项目目录操作](#项目目录操作)
4. [数据处理命令](#数据处理命令)
5. [模型训练命令](#模型训练命令)
6. [Docker 容器管理](#docker-容器管理)
7. [系统监控命令](#系统监控命令)
8. [常用快捷键](#常用快捷键)

---

## Python 环境管理

### 创建虚拟环境

```bash
# Windows
python -m venv venv

# Linux/Mac
python3 -m venv venv
```

### 激活虚拟环境

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 退出虚拟环境

```bash
deactivate
```

### 安装依赖包

```bash
# 安装单个包
pip install numpy

# 从 requirements.txt 安装所有依赖
pip install -r requirements.txt

# 指定版本安装
pip install numpy==1.21.0

# 升级包
pip install --upgrade numpy

# 卸载包
pip uninstall numpy
```

### 查看已安装的包

```bash
# 列出所有已安装的包
pip list

# 显示包的详细信息
pip show numpy

# 导出依赖包列表
pip freeze > requirements.txt
```

### 运行 Python 脚本

```bash
# 运行脚本
python script.py

# 运行脚本并传递参数
python script.py --arg1 value1 --arg2 value2
```

---

## Git 版本控制

### 初始化仓库

```bash
# 初始化本地仓库
git init

# 克隆远程仓库
git clone https://github.com/username/repository.git
```

### 查看状态和日志

```bash
# 查看当前状态
git status

# 查看提交历史
git log

# 查看简洁的提交历史
git log --oneline

# 查看文件修改内容
git diff

# 查看特定文件的修改
git diff filename.py
```

### 添加和提交

```bash
# 添加所有修改的文件
git add .

# 添加特定文件
git add filename.py

# 提交变更
git commit -m "提交信息"

# 添加并提交
git commit -am "提交信息"
```

### 分支管理

```bash
# 查看所有分支
git branch

# 创建新分支
git branch branch-name

# 切换到指定分支
git checkout branch-name

# 创建并切换到新分支
git checkout -b branch-name

# 合并分支
git merge branch-name

# 删除分支
git branch -d branch-name
```

### 远程仓库操作

```bash
# 添加远程仓库
git remote add origin https://github.com/username/repository.git

# 查看远程仓库
git remote -v

# 推送到远程仓库
git push origin master

# 拉取远程仓库的更新
git pull origin master

# 获取远程仓库的更新但不合并
git fetch origin
```

### 标签管理

```bash
# 创建标签
git tag v1.0.0

# 查看所有标签
git tag

# 删除标签
git tag -d v1.0.0

# 推送标签到远程仓库
git push origin v1.0.0
```

### 撤销操作

```bash
# 撤销工作区的修改
git checkout -- filename.py

# 撤销暂存区的修改
git reset HEAD filename.py

# 撤销最后一次提交
git reset --soft HEAD~1

# 撤销最后一次提交并保留修改
git reset --mixed HEAD~1

# 撤销最后一次提交并丢弃修改
git reset --hard HEAD~1

# 回滚到指定提交
git revert commit-hash
```

### 提交信息规范

```bash
# 格式：<type>(<scope>): <subject>

# 类型说明：
feat:     新功能
fix:      修复 bug
docs:     文档更新
style:    代码格式调整
refactor: 重构代码
test:     测试相关
chore:    构建过程或辅助工具变动

# 示例：
git commit -m "feat: 添加水位计算模块"
git commit -m "fix: 修复图表中文乱码问题"
git commit -m "docs: 更新 README 文档"
```

---

## 项目目录操作

### 创建目录

```bash
# 创建单个目录
mkdir src

# 创建多级目录
mkdir -p src/modules

# 创建多个目录
mkdir src data output
```

### 查看目录内容

```bash
# 查看当前目录内容
ls

# 查看详细信息
ls -l

# 查看隐藏文件
ls -a

# 查看文件大小
ls -lh

# Windows
dir
dir /a
```

### 切换目录

```bash
# 切换到指定目录
cd src

# 返回上一级目录
cd ..

# 返回上两级目录
cd ../..

# 返回用户主目录
cd ~

# Windows
cd \
```

### 删除目录

```bash
# 删除空目录
rmdir directory-name

# 删除非空目录及其内容
rm -rf directory-name

# Windows
rmdir directory-name
rd /s /q directory-name
```

### 复制和移动

```bash
# 复制文件
cp file1.py file2.py

# 复制目录
cp -r src src-backup

# 移动文件
mv file1.py file2.py

# 移动目录
mv src new-src

# Windows
copy file1.py file2.py
xcopy /E /I src src-backup
move file1.py file2.py
```

### 查找文件

```bash
# 查找文件
find . -name "*.py"

# 查找目录
find . -type d -name "src"

# 在文件中查找内容
grep "keyword" filename.py

# 在多个文件中查找内容
grep -r "keyword" ./

# Windows
dir /s /b *.py
findstr /s /i "keyword" *.py
```

---

## 数据处理命令

### 查看数据文件

```bash
# 查看 CSV 文件前 10 行
head -n 10 data.csv

# 查看 CSV 文件后 10 行
tail -n 10 data.csv

# 实时查看文件内容
tail -f log.txt

# 查看 Excel 文件（需要安装工具）
in2csv data.xlsx | head -n 10
```

### 数据转换

```bash
# Excel 转 CSV
in2csv data.xlsx > data.csv

# CSV 转 Excel
csv2excel data.csv data.xlsx

# JSON 转 CSV
in2csv data.json > data.csv
```

### 数据压缩和解压

```bash
# 压缩文件
tar -czf archive.tar.gz directory/

# 解压文件
tar -xzf archive.tar.gz

# 压缩为 zip
zip -r archive.zip directory/

# 解压 zip
unzip archive.zip

# Windows
tar -czf archive.tar.gz directory/
tar -xzf archive.tar.gz
```

---

## 模型训练命令

### 启动 Jupyter Notebook

```bash
# 启动 Jupyter Notebook
jupyter notebook

# 指定端口启动
jupyter notebook --port 8888

# 后台运行
nohup jupyter notebook &

# 启动 Jupyter Lab
jupyter lab
```

### 训练模型

```bash
# 训练 LSTM 模型
python train_lstm.py --epochs 100 --batch_size 32

# 训练 KNN 模型
python train_knn.py --k 5 --metric euclidean

# 使用 GPU 训练
CUDA_VISIBLE_DEVICES=0 python train_model.py
```

### 模型评估

```bash
# 评估模型
python evaluate.py --model best_model.h5 --data test_data.csv

# 生成预测结果
python predict.py --model best_model.h5 --input data.csv --output predictions.csv
```

### 模型部署

```bash
# 启动 Flask 服务
python app.py --host 0.0.0.0 --port 5000

# 使用 Gunicorn 部署
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Docker 部署
docker-compose up -d
```

---

## Docker 容器管理

### 构建镜像

```bash
# 构建镜像
docker build -t smart-water:latest .

# 使用 Dockerfile 构建
docker build -f Dockerfile -t smart-water:latest .

# 查看镜像
docker images
```

### 运行容器

```bash
# 运行容器
docker run -d -p 5000:5000 --name smart-water smart-water:latest

# 查看运行中的容器
docker ps

# 查看所有容器
docker ps -a

# 查看容器日志
docker logs smart-water

# 进入容器
docker exec -it smart-water /bin/bash
```

### 容器管理

```bash
# 停止容器
docker stop smart-water

# 启动容器
docker start smart-water

# 重启容器
docker restart smart-water

# 删除容器
docker rm smart-water

# 删除镜像
docker rmi smart-water:latest
```

### Docker Compose

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 查看服务状态
docker-compose ps

# 查看服务日志
docker-compose logs -f

# 重启服务
docker-compose restart
```

---

## 系统监控命令

### CPU 和内存监控

```bash
# 查看系统资源使用情况
top

# 实时查看 CPU 使用情况
htop

# 查看 CPU 信息
lscpu

# 查看内存使用情况
free -h

# Windows
tasklist
taskmgr
```

### 磁盘管理

```bash
# 查看磁盘使用情况
df -h

# 查看目录大小
du -sh directory/

# 查看目录下各子目录大小
du -sh */

# Windows
dir
wmic logicaldisk get size,freespace,caption
```

### 进程管理

```bash
# 查看进程
ps aux

# 查找特定进程
ps aux | grep python

# 杀死进程
kill PID

# 强制杀死进程
kill -9 PID

# Windows
tasklist
taskkill /PID PID /F
```

### 网络管理

```bash
# 查看网络连接
netstat -tuln

# 查看端口占用
netstat -tuln | grep 5000

# 测试网络连接
ping google.com

# 测试端口连通性
telnet localhost 5000

# Windows
netstat -an
ping google.com
```

---

## 常用快捷键

### 终端快捷键

```bash
Ctrl + C      # 中断当前命令
Ctrl + Z      # 暂停当前命令
Ctrl + D      # 退出当前 shell
Ctrl + L      # 清屏
Ctrl + A      # 移动到行首
Ctrl + E      # 移动到行尾
Ctrl + U      # 删除光标前所有字符
Ctrl + K      # 删除光标后所有字符
Ctrl + W      # 删除光标前一个单词
Ctrl + R      # 搜索历史命令
!!            # 执行上一条命令
!$            # 上一条命令的最后一个参数
```

### Vim 快捷键

```bash
i             # 进入插入模式
Esc           # 退出插入模式
:w            # 保存
:q            # 退出
:wq           # 保存并退出
:q!           # 强制退出不保存
dd            # 删除当前行
yy            # 复制当前行
p             # 粘贴
/keyword      # 搜索关键词
n             # 下一个匹配
N             # 上一个匹配
```

---

## 常见问题解决

### Git 冲突解决

```bash
# 1. 查看冲突文件
git status

# 2. 打开冲突文件，查找冲突标记 <<<<<<<

# 3. 手动编辑文件，解决冲突

# 4. 标记冲突已解决
git add filename.py

# 5. 提交合并
git commit -m "fix: 解决合并冲突"
```

### Python 包安装失败

```bash
# 使用清华镜像源安装
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy

# 升级 pip
pip install --upgrade pip

# 使用 root 权限安装（Linux）
sudo pip install numpy

# 清除 pip 缓存
pip cache purge
```

### 端口被占用

```bash
# 查看端口占用
lsof -i:5000
# 或
netstat -tuln | grep 5000

# 杀死占用端口的进程
kill -9 PID

# Windows
netstat -ano | findstr :5000
taskkill /PID PID /F
```

---

## 总结

本手册涵盖了智慧水利项目开发中常用的 Bash 命令，包括：

- ✅ Python 环境管理
- ✅ Git 版本控制
- ✅ 项目目录操作
- ✅ 数据处理命令
- ✅ 模型训练命令
- ✅ Docker 容器管理
- ✅ 系统监控命令
- ✅ 常用快捷键

**建议**：将本手册保存为书签，方便随时查阅。

---

**最后更新**：2026年1月24日
