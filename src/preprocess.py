import sys
from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pypdf import PdfReader
import time

# ==================== EPUB 提取 ====================
def extract_epub(epub_path):
    """从 EPUB 提取纯文本"""
    book = epub.read_epub(epub_path)
    full_text = ""
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            try:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text = soup.get_text()
                if text.strip():
                    full_text += text + "\n\n"
            except:
                continue
    
    return full_text

# ==================== PDF 提取 ====================
def extract_pdf(pdf_path):
    """从 PDF 提取纯文本（文字版）"""
    reader = PdfReader(pdf_path)
    full_text = ""
    total_pages = len(reader.pages)
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
        # 每10页显示一次进度
        if page_num % 10 == 0:
            print(f"      已提取 {page_num}/{total_pages} 页")
    
    return full_text

# ==================== 文件大小提示 ====================
def get_file_size_mb(path):
    return path.stat().st_size / (1024 * 1024)

# ==================== 主程序 ====================
if __name__ == "__main__":
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集 EPUB 和 PDF 文件
    files = list(raw_dir.glob("*.epub")) + list(raw_dir.glob("*.pdf"))
    
    if not files:
        print("❌ data/raw 中没有找到 EPUB 或 PDF 文件")
        sys.exit(1)
    
    print(f"📂 找到 {len(files)} 个文件\n")
    print("=" * 50)
    
    for file_path in files:
        suffix = file_path.suffix.lower()
        output_name = file_path.stem + ".txt"
        output_path = processed_dir / output_name
        file_size = get_file_size_mb(file_path)
        
        # 跳过已处理的（文件 > 1KB 视为已处理）
        if output_path.exists() and output_path.stat().st_size > 1024:
            print(f"⏭️  跳过: {file_path.name} (已存在)")
            continue
        
        print(f"\n📖 正在处理: {file_path.name}")
        print(f"   📦 文件大小: {file_size:.1f} MB")
        
        start_time = time.time()
        
        try:
            # 根据后缀选择提取方法
            if suffix == ".epub":
                text = extract_epub(file_path)
            elif suffix == ".pdf":
                text = extract_pdf(file_path)
            else:
                print(f"   ⚠️ 不支持格式: {suffix}，跳过")
                continue
            
            elapsed = time.time() - start_time
            
            if text and len(text.strip()) > 100:
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                char_count = len(text)
                print(f"   ✅ 提取成功!")
                print(f"      📝 字符数: {char_count:,}")
                print(f"      ⏱️  耗时: {elapsed:.1f} 秒")
                print(f"      💾 保存至: {output_path}")
            else:
                print(f"   ⚠️ 文本为空或过短，跳过")
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    print("\n" + "=" * 50)
    print("\n🎉 预处理完成！")
    print(f"📂 所有结果保存在: {processed_dir.absolute()}")
    
    # 显示结果统计
    txt_files = list(processed_dir.glob("*.txt"))
    if txt_files:
        print("\n📊 已生成的文件:")
        for txt in sorted(txt_files):
            size_kb = txt.stat().st_size / 1024
            print(f"   📄 {txt.name} ({size_kb:.1f} KB)")