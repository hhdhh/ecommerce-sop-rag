"""
生成 200 条跨境电商 SOP 到 data/sops/，覆盖：
  - 亚马逊 (Amazon)
  - TikTok Shop
  - Shopee
  - Temu
  - 通用 (跨平台)

业务线：
  - Listing 优化
  - 流量与广告
  - 售后与客服
  - 知识产权与合规
  - 物流与仓储
  - 财税与合规
  - 选品与市场调研

每条 SOP 含 frontmatter + 多个二级章节。
"""
from __future__ import annotations

import json
import os
import random
import textwrap
from pathlib import Path

random.seed(42)

OUT = Path(__file__).resolve().parent.parent / "data" / "sops"
OUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 业务线 / 场景库（按真实跨境电商业务结构设计）
# =============================================================================

PLATFORMS = ["Amazon", "TikTok Shop", "Shopee", "Temu", "通用"]
CATEGORIES = [
    "Listing 优化",
    "流量与广告",
    "售后与客服",
    "知识产权与合规",
    "物流与仓储",
    "财税与合规",
    "选品与市场调研",
]

# 每个业务线下的 SOP 模板
TEMPLATES: dict[str, list[dict]] = {
    "Listing 优化": [
        {
            "title": "{platform} Listing 标题与五点描述 SOP",
            "sections": [
                ("标题结构", "## 标题结构\n\n1. 品牌名 + 核心关键词 + 属性词 + 适用场景 + 容量/规格\n2. 字符限制：{char_limit}\n3. 禁止堆砌：同一关键词最多出现 2 次\n4. 禁止使用：best、#1、guaranteed 等绝对化用词"),
                ("五点描述", "## 五点描述\n\n每点 200 字符以内，按以下顺序：\n1. 核心卖点（材质/工艺）\n2. 使用场景（解决什么问题）\n3. 差异化优势（对比同行）\n4. 包装清单\n5. 售后承诺"),
                ("A+ 页面", "## A+ 页面布局\n\n- 模块 1：品牌故事 + 高对比场景图\n- 模块 2：3 张对比图（功能拆解）\n- 模块 3：场景化使用流程图\n- 模块 4：FAQ 模块（前 5 个常见问题）\n- 模块 5：跨链接模块（关联 3 个互补品）"),
                ("关键词策略", "## 关键词策略\n\n1. 核心词：搜索量 > 5000，置于标题前 80 字符\n2. 长尾词：搜索量 500-5000，分布于五点描述\n3. 竞品词：监控 top 20 ASIN 标题词频，差异化补全\n4. 反查工具：Helium 10 / Jungle Scout / Cerebro"),
                ("合规红线", "## 合规红线\n\n- 禁用：未授权的品牌名、名人姓名\n- 禁用：医疗/减肥/金融功效承诺\n- 禁用：与图片不符的型号/参数\n- 侵权自查：上架前 24h 用工具扫描商标局数据库"),
            ],
        },
        {
            "title": "{platform} 主图与视频 SOP",
            "sections": [
                ("主图要求", "## 主图要求\n\n- 纯白底（RGB 255,255,255），商品占比 85% 以上\n- 尺寸：{img_size}，最小 1000x1000 像素\n- 不能含文字、徽章、促销信息（亚马逊规定）\n- 阴影/倒影：可选，建议移除以提升一致性"),
                ("辅图规划", "## 辅图规划（7 张）\n\n1. 卖点 1：材质/工艺特写\n2. 卖点 2：尺寸/参数示意\n3. 卖点 3：使用场景\n4. 卖点 4：对比图（vs 同行）\n5. 卖点 5：包装清单\n6. 卖点 6：安装/使用步骤\n7. 信任背书：认证标识 / 用户好评截图"),
                ("视频脚本", "## 视频脚本（30 秒）\n\n0-5s：痛点 hook（3 个字字幕）\n5-15s：产品展示 + 核心卖点字幕\n15-25s：使用场景 + 对比\n25-30s：CTA + 价格 + 折扣码"),
                ("图片 SEO", "## 图片 SEO\n\n- 文件名：英文关键词-数字.jpg（amazon 搜索引擎会读 alt）\n- Alt 文本：核心关键词 + 1-2 个属性词\n- 压缩：< 200KB 用 TinyPNG / Squoosh"),
            ],
        },
    ],
    "流量与广告": [
        {
            "title": "{platform} 站内广告投放 SOP",
            "sections": [
                ("广告结构", "## 广告结构（3 层）\n\n- 自动广告：跑词 + 收集数据，预算 30%\n- 手动广泛：拓词，预算 40%\n- 手动精准：收割高转化词，预算 30%"),
                ("出价策略", "## 出价策略\n\n1. 新品期（前 14 天）：建议出价 = 竞品首页 CPC × 1.2\n2. 成长期（14-60 天）：根据 ACoS 调整，目标 ACoS < 25%\n3. 成熟期：分层管理，核心词提高出价抢首位，长尾词降价"),
                ("关键词否定", "## 关键词否定\n\n- 每周一查 search term report\n- 否定标准：曝光 > 1000 且 CTR < 0.2% 且 CVR < 5%\n- 否定类型：精准否定（不相关词）/ 词组否定（广泛误匹配）"),
                ("分时调整", "## 分时调整\n\n- 拉取过去 30 天分时销售数据\n- 0-7 点降低出价 30%（流量低，ACoS 高）\n- 19-23 点提高出价 20%（购物高峰）"),
            ],
        },
        {
            "title": "{platform} 站外引流 SOP",
            "sections": [
                ("达人合作", "## 达人合作\n\n- 选号标准：粉丝 5k-100k，互动率 > 3%\n- 沟通模板：brief 含核心卖点 + 必出镜头 + 禁用话术\n- 报价区间：1k-10k 粉：$30-150；10k-100k：$150-800\n- 验收：链接 + 完播率截图 + 30 天数据复盘"),
                ("Deal 站点", "## Deal 站点\n\n- 主流：Slickdeals、Dealnews、Reddit r/deals\n- 节奏：每月最多 2 次，每次折扣深度 > 25%\n- 风险：BD 词违规（点击诱导词）会被限流"),
                ("社媒矩阵", "## 社媒矩阵\n\n- Instagram：产品图 + Reels（15s 短视频）\n- TikTok：原生感 + 槽点 + 测评（不要硬广）\n- Facebook Group：行业垂直群，资源置换引流\n- YouTube：长测评（5-10 分钟），SEO 关键词布局"),
            ],
        },
    ],
    "售后与客服": [
        {
            "title": "{platform} 退换货处理 SOP",
            "sections": [
                ("退货原因分类", "## 退货原因分类\n\n1. 与描述不符：差评风险高，48h 内联系买家\n2. 损坏/质量问题：直接补发 + 道歉，零退货\n3. 不想要了：标准流程，30 天内可退\n4. 物流慢/丢件：核实后全额退款 + 留资"),
                ("退款模板", "## 退款模板（中英双语）\n\n中文：非常抱歉给您带来困扰，您的退款已通过原渠道退回，预计 3-5 个工作日到账。本次给您额外补偿 5% 店铺积分，期待您再次光临。\n\n英文：We sincerely apologize for the inconvenience. Your full refund has been issued via the original payment method and will arrive within 3-5 business days. As compensation, we've credited an extra 5% store credit to your account."),
                ("Return Reason 监控", "## Return Reason 监控\n\n- 每周拉取退货原因 top 10\n- 客诉 > 5% 的 SKU 触发 listing 下架复审\n- 同一供应商 SKU 集中投诉 → 启动供应商审查"),
            ],
        },
        {
            "title": "{platform} 差评应对 SOP",
            "sections": [
                ("差评分级", "## 差评分级\n\n- S 级：1-2 星 + 图片 + 详细描述 → 24h 响应\n- A 级：1-2 星 + 短描述 → 48h 响应\n- B 级：3 星 + 改进建议 → 72h 响应\n- C 级：买家秀差评 → 主动联系求删/改评"),
                ("回复模板", "## 回复模板（公开 + 私信）\n\n公开：Dear customer, thank you for your feedback. We take this very seriously and would love to make it right. Please contact us at support@brand.com with your order number.\n\n私信：补充解决方案 + 退款/补发 + 主动求删/改评"),
                ("合规话术", "## 合规话术\n\n- 不允许：诱导删评（违反 Amazon TOS）\n- 不允许：威胁买家\n- 允许：诚恳道歉 + 主动提供解决方案 + 引导线下沟通"),
                ("根因分析", "## 根因分析\n\n每周差评聚类（NLP 主题分类）：\n- 物流类 → 物流团队\n- 质量类 → 品控团队\n- 描述不符 → Listing 团队\n- 客服态度 → 客服团队"),
            ],
        },
    ],
    "知识产权与合规": [
        {
            "title": "{platform} 知识产权侵权申诉 SOP",
            "sections": [
                ("收到投诉后 24h 行动", "## 收到投诉后 24h 行动\n\n1. 拉取投诉邮件原文，记录：投诉方、投诉商标号、投诉理由、ASIN 清单\n2. 内部初步判断：是否真侵权（核对自家产品供应链文件）\n3. 真侵权：立即下架 + 库存处理（销毁 / 改包装）\n4. 假投诉 / 恶意投诉：进入申诉流程"),
                ("申诉材料准备", "## 申诉材料准备\n\n- 商标查询截图（USPTO / 当地商标局）\n- 供应商授权链（采购合同 + 商标使用授权）\n- 产品实拍图（与投诉方对比，证明不构成混淆）\n- 历史销售记录（证明上架时间早于投诉方注册时间）\n- 律师信（专业案件）"),
                ("申诉信模板", "## 申诉信模板\n\n尊敬的 {platform} 团队：\n\n我们就 [ASIN 列表] 被 [投诉方] 以 [商标号] 投诉侵权一案提出申诉。\n\n事实与理由：\n1. 我司 [商标号] 在 [国家] 已于 [日期] 注册，早于投诉方 [日期]\n2. 我司产品采用 [自有设计]，与投诉方产品在 [外观 / 名称 / 功能] 上存在显著差异，不会造成消费者混淆\n3. 附件：[证据清单]\n\n请贵平台撤销本次投诉。如需进一步信息，请联系法务邮箱 legal@brand.com。\n\n此致\n[公司名] 法务部\n[日期]"),
                ("预防机制", "## 预防机制\n\n- 新品上架前：商标库 + 外观专利库双查\n- 监控：每周扫一次投诉高发关键词\n- 供应链：要求供应商提供完整授权链文件"),
            ],
        },
    ],
    "物流与仓储": [
        {
            "title": "{platform} FBA 备货 SOP",
            "sections": [
                ("销量预测", "## 销量预测\n\n公式：备货量 = (日均销量 × 备货周期) + 安全库存\n\n- 日均销量：取过去 60 天加权平均\n- 备货周期：头程海运 35 天 + 清关 5 天 + 入库 7 天 = 47 天\n- 安全库存：日均 × 14 天（应对爆单 / 断货风险）"),
                ("头程选择", "## 头程选择\n\n- 海运（美森快船）：35 天，成本最低，旺季慎用\n- 海运（普船）：45-55 天，超大件首选\n- 空运：7-10 天，紧急补货用，单位成本 4-6 倍\n- 快递（UPS/FedEx）：5-7 天，小批量补货"),
                ("库存监控", "## 库存监控\n\n- 周更：可售天数 = 当前 FBA 库存 / 日均销量\n- 红灯：< 14 天 → 立即补货\n- 黄灯：14-30 天 → 启动下一批备货\n- 绿灯：> 30 天 → 评估是否调价清库存"),
            ],
        },
    ],
    "财税与合规": [
        {
            "title": "{platform} 欧洲 VAT 注册与申报 SOP",
            "sections": [
                ("注册触发条件", "## 注册触发条件\n\n- 欧盟 B2C 商品销售额超过远程销售阈值（€10,000/年）\n- 在某一成员国仓储（FBA仓 / 海外仓）\n- 主动选择：在哪个国家仓储就在哪个国家注册"),
                ("申报周期", "## 申报周期\n\n- 季度申报：每个季度结束后次月申报\n- 年度申报：次年 1-3 月\n- 申报内容：销售额、采购进项、应缴 / 应退 VAT"),
                ("IOSS / OSS", "## IOSS / OSS\n\n- IOSS：货值 ≤ €150 的 B2C 进口小包用，简化清关\n- OSS（One-Stop-Shop）：欧盟内远程销售汇总申报，避免多国注册"),
            ],
        },
    ],
    "选品与市场调研": [
        {
            "title": "{platform} 选品调研 SOP",
            "sections": [
                ("市场容量分析", "## 市场容量分析\n\n- 核心类目月销量 > 100k 件\n- top 100 ASIN 平均月销 > 500 件\n- 类目增长率（YoY）> 10%"),
                ("竞争强度", "## 竞争强度\n\n- 头部品牌集中度：top 10 份额 < 60% 为佳\n- 评论壁垒：top 100 平均评论数 < 3000 进入门槛低\n- 专利 / 品牌备案：扫描类目是否有大量防御性商标"),
                ("利润测算", "## 利润测算\n\n售价 - 采购 - 头程 - FBA - 平台费 - 广告 - 退货 - VAT = 净利润\n目标净利润率 > 15%"),
                ("合规检查", "## 合规检查\n\n- 类目是否需要类目审核（如美妆、食品）\n- 是否有强制认证（CE / FCC / FDA）\n- 平台合规：图片 / 标题 / 关键词是否有违规风险"),
            ],
        },
    ],
}


# =============================================================================
# 生成器
# =============================================================================

def _gen_one(idx: int, platform: str, category: str, tpl: dict) -> str:
    char_limit = {
        "Amazon": "200 字符",
        "TikTok Shop": "40 字符",
        "Shopee": "100 字符",
        "Temu": "60 字符",
        "通用": "120 字符",
    }[platform]
    img_size = {
        "Amazon": "2000x2000",
        "TikTok Shop": "1080x1080",
        "Shopee": "1000x1000",
        "Temu": "1200x1200",
        "通用": "1080x1080",
    }[platform]

    title = tpl["title"].format(platform=platform, char_limit=char_limit, img_size=img_size)
    sop_id = f"SOP-{category[:2]}-{platform[:3].upper()}-{idx:04d}"

    frontmatter = textwrap.dedent(f"""\
        ---
        id: {sop_id}
        title: {title}
        platform: {platform}
        category: {category}
        version: 1.{random.randint(0, 9)}.{random.randint(0, 9)}
        owner: ops@brand.com
        last_updated: 2026-06-10
        ---

        # {title}

        > 适用范围：{platform} {category} 业务
        > 责任人：{random.choice(['张磊', '李静', '王浩', '陈晨', '刘洋', '赵敏'])}（{category}组）
        > 评审：周更

        """)

    body_parts: list[str] = []
    for section_name, section_body in tpl["sections"]:
        # 模板里如果已经有 ## 标题就不重复加
        body_clean = section_body
        if not body_clean.lstrip().startswith("## "):
            body_clean = f"## {section_name}\n\n{body_clean}"
        # 处理占位符
        try:
            body_clean = body_clean.format(
                platform=platform,
                char_limit=char_limit,
                img_size=img_size,
            )
        except KeyError:
            pass
        body_parts.append(body_clean)
    body = "\n\n".join(body_parts)
    return frontmatter + body + "\n"


def generate_all(target_count: int = 200) -> list[Path]:
    out_files: list[Path] = []
    counter = 0
    # 轮询：每个类目分配合理权重
    weights = {
        "Listing 优化": 0.20,
        "流量与广告": 0.18,
        "售后与客服": 0.18,
        "知识产权与合规": 0.12,
        "物流与仓储": 0.12,
        "财税与合规": 0.08,
        "选品与市场调研": 0.12,
    }
    for category, weight in weights.items():
        n = int(target_count * weight)
        templates = TEMPLATES[category]
        for i in range(n):
            platform = PLATFORMS[i % len(PLATFORMS)]
            tpl = templates[i % len(templates)]
            counter += 1
            content = _gen_one(counter, platform, category, tpl)
            fname = f"{category[:2]}-{platform[:3]}-{counter:04d}.md"
            path = OUT / fname
            path.write_text(content, encoding="utf-8")
            out_files.append(path)

    return out_files


if __name__ == "__main__":
    files = generate_all(200)
    print(f"✅ 生成 {len(files)} 条 SOP 到 {OUT}")
    # manifest
    manifest = [
        {
            "file": p.name,
            "size": p.stat().st_size,
            "category": p.name.split("-")[0],
        }
        for p in files
    ]
    (OUT.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  manifest.json 已更新")
