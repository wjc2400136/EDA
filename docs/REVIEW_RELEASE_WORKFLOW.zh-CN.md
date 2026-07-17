# 审稿期与公开发布流程

[English](REVIEW_RELEASE_WORKFLOW.md) | [简体中文](REVIEW_RELEASE_WORKFLOW.zh-CN.md)

## 应该上传哪个目录

不要把作者维护的 `EDA_reproducibility_package/` 原目录直接交给匿名服务。该目录是
接收后公开版本的源目录，`LICENSE`、`CITATION.cff`、`pyproject.toml` 和第三方声明中
有意保留了真实作者元数据。

应先生成审稿产物：

```bash
python tools/build_anonymous_snapshot.py --force
```

该命令生成：

```text
dist/
|-- eda-review-anonymous/             # 匿名审稿仓库应上传的内容
|-- eda-review-anonymous.zip          # 可上传到 Editorial Manager 的补充代码
|-- eda-review-anonymous.zip.sha256   # 固定校验值
`-- UPLOAD_INSTRUCTIONS.txt
```

匿名仓库只使用 `dist/eda-review-anonymous/` **里面的内容**；随稿补充材料使用
`dist/eda-review-anonymous.zip`。生成器会排除数据集、checkpoint、实验输出、服务商
响应、API 密钥、作者元数据和个人路径，同时生成审稿人快速入门文档并扫描身份泄漏。

## 推荐的匿名只读方案

只有作者可访问的 Private repository 不能单独满足审稿人的代码访问要求。推荐流程为：

1. 在 GitHub 新建一个专门用于匿名快照的空 **Private repository**，不要让 GitHub
   额外初始化 README、license 或 `.gitignore`。
2. 只提交 `dist/eda-review-anonymous/` 中的内容。不要沿用作者工作仓库的 Git 历史，
   也不要把匿名文件放到会显示作者身份的仓库中。
3. 冻结一个 commit，例如 `review-v1`，并记录 commit SHA 和 ZIP 的 SHA-256。
4. 在 [Anonymous GitHub](https://anonymous.4open.science/) 中基于这个 private source
   repository 创建只读镜像。优先固定到具体 commit，并在审稿期间关闭自动更新。
5. 在服务的自定义脱敏词中加入全部作者姓名、单位、邮箱、GitHub 用户名或组织名、
   private repository 名称和有辨识度的本地项目路径。这些词不要写入匿名仓库本身。
6. 关闭或逐项检查可能泄露身份的外部链接和二进制文件。本快照默认不含论文 PDF、
   生成图、数据集、权重和服务商原始响应。
7. 过期时间应晚于预计审稿周期；审稿期间到期操作应选择删除内容，不要重定向到
   暴露作者身份的 GitHub 仓库。
8. 使用退出登录的浏览器打开匿名 URL，下载归档，并在下载副本中运行
   `python tools/check_release.py`。

Anonymous GitHub 提供的是去身份、只读镜像，不会把 private source repository 直接
公开。镜像私有源仓库时需要授权该服务读取 GitHub 仓库；授权前应再次查看其当前权限
和保留策略。如果不能接受该授权，则改用生成的 ZIP 作为 Editorial Manager 补充代码。

如果仓库超过服务的直接下载大小限制，应使用其 proxy/stream 模式或删减非必要资源。
无论是否使用匿名镜像，都应保留生成的 ZIP 和 SHA-256，作为固定且可核验的备份。

## 创建专用匿名审稿仓库

从生成的匿名目录建立全新的 Git 仓库：

```bash
cd dist/eda-review-anonymous
git init
git branch -M main
git add .
git status --short
git commit -m "Anonymous review snapshot review-v1"
git remote add origin https://github.com/OWNER/PRIVATE_REVIEW_REPOSITORY.git
git push -u origin main
git rev-parse HEAD
```

创建匿名审稿仓库时，不要在 `EDA_reproducibility_package/` 根目录执行上述命令。该根
目录只适合作为作者 private 开发仓库或接收后的公开仓库。

## 提交前检查

1. 确认 `REVIEWER_QUICKSTART.md` 可正常打开，且命令均从仓库根目录运行。
2. 确认不存在数据图像、权重、对抗输出、VLM 原始响应、密钥、个人路径、作者姓名、
   致谢或单位信息。
3. 退出登录后验证匿名 URL，且审稿人无需向作者邮箱申请权限。
4. 将上传 ZIP 的哈希与 `eda-review-anonymous.zip.sha256` 对比。
5. 审稿期间不修改冻结 commit；如必须修正，创建 `review-v2`，记录修改内容，并同步
   更新匿名链接和补充压缩包。
6. 当代码只存在于 private source repository 或匿名审稿链接时，不要声称代码已经
   publicly available。

## 回复信建议措辞

提交前用已测试的真实匿名 URL、commit 和哈希替换占位符，只保留实际提供的渠道。

```tex
\noindent\textbf{Response:}
Thank you for this suggestion. We have prepared a complete reproducibility
package containing the EDA implementation, attack-generation and evaluation
scripts, environment specifications, dataset and checkpoint preparation
instructions, random seeds, and detailed documentation. The identity-scrubbed
snapshot used for the revised experiments is available to the reviewers through
the read-only anonymous repository at \url{<ANONYMOUS_REVIEW_URL>}. The same
snapshot is provided as \texttt{<SUPPLEMENTARY_ARCHIVE_NAME>} and is identified
by commit \texttt{<REVIEW_COMMIT>} and SHA-256 \texttt{<ARCHIVE_SHA256>}.
Dataset images, third-party model weights, generated outputs, provider responses,
and API credentials are not redistributed. Upon acceptance, we will publish the
reviewed snapshot in a versioned repository and create a persistent archival
release.
```

## 审稿阶段论文中的 Code Availability

```tex
\section*{Code availability}
The complete code and reproducibility package for the revised experiments are
available to reviewers through a read-only anonymous repository and the
supplementary code archive. The package includes environment specifications,
data and checkpoint preparation instructions, attack-generation and evaluation
scripts, random seeds, and expected output formats. The reviewed snapshot will
be made publicly available in a versioned archival release upon acceptance.
```

## 接收后公开

论文接收后公开本次审稿快照，为论文版本打标签（例如 `v1.0.0`），创建 GitHub
Release，并通过可提供持久 DOI 的服务归档。随后更新 `CITATION.cff`、README 和论文
中的公开 URL 与 DOI。数据集和第三方模型权重继续只提供准备说明，不打包分发。

