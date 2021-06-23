import os

for md_file in os.listdir('./notes/src/'):
    fileName = str(md_file).replace('.md','')
    os.system(f'pandoc ./notes/src/{md_file} --pdf-engine=xelatex -o ./notes/src/{fileName}.pdf -V geometry:margin=1in')
    os.system(f'mv ./notes/src/{fileName}.pdf ./notes/pdf/{fileName}.pdf')