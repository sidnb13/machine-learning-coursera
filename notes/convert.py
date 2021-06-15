import os

for md_file in os.listdir('./notes/src/'):
    fileName = str(md_file).replace('.md','')
    os.system(f'pdfmd ./notes/src/{md_file}')
    os.system(f'mv ./notes/src/{fileName}.pdf ./notes/pdf/{fileName}.pdf')