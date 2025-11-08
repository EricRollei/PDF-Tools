import fitz

doc = fitz.open(r'C:\Users\Eric\Downloads\lauren_stevenson_portfolio.pdf')
over_250 = 0
all_sizes = []

for page_num in range(len(doc)):
    page = doc[page_num]
    imgs = page.get_images()
    
    print(f'\nPage {page_num+1}: {len(imgs)} images')
    for i, img in enumerate(imgs):
        width = img[2]
        height = img[3]
        min_side = min(width, height)
        all_sizes.append((page_num+1, width, height, min_side))
        
        meets_criteria = min_side >= 250
        if meets_criteria:
            over_250 += 1
            print(f'  ✅ Image {i+1}: {width}x{height} px (min: {min_side})')
        else:
            print(f'  ❌ Image {i+1}: {width}x{height} px (min: {min_side}) - TOO SMALL')

print(f'\n{"="*60}')
print(f'Total images in PDF: {len(all_sizes)}')
print(f'Images >= 250px on smallest side: {over_250}')
print(f'Images < 250px (will be filtered): {len(all_sizes) - over_250}')
print(f'{"="*60}')

doc.close()
