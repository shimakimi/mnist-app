const canvasElement = document.getElementById('draw-area')  
const canvas = new HandwritingCanvas(canvasElement) 

const clearButtonElement = document.getElementById('clear-button')
clearButtonElement.addEventListener('click', () => {
    canvas.clear()
})

const predictButtonElement = document.getElementById('predict-button')
predictButtonElement.addEventListener('click', async () => {
    if (canvas.isEmpty) {
        return
    }

    // 推論実行

    const blob = await canvas.toBlob('image/png')
    const formData = new FormData()
    formData.append('image', blob, 'number.png')

    const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    const responseData = await response.json()

    // 推論結果の画像を表示

    const imageUrl = URL.createObjectURL(blob)

    const imageElement = document.createElement('img')
    imageElement.src = imageUrl

    const resultImageElement = document.getElementById('result-image')

    if (resultImageElement.firstChild) {
        resultImageElement.removeChild(resultImageElement.firstChild)
    }

    resultImageElement.appendChild(imageElement)

    canvas.clear()
    
    // 推論結果をtbodyに表示

    const tableBodyElement = document.getElementById('result-table-body')

    while (tableBodyElement.firstChild) {
        tableBodyElement.removeChild(tableBodyElement.firstChild)
    }

    const probabilities = responseData.probabilities
    for (let i = 0; i < probabilities.length; i++) {
        const tr = document.createElement('tr')

        // 数字
        const tdNumber = document.createElement('td')
        tdNumber.textContent = i
        tr.appendChild(tdNumber)

        //確率
        const tdProbability = document.createElement('td')
        tdProbability.textContent = (probabilities[i] * 100).toFixed(1)
        tr.appendChild(tdProbability)

        tableBodyElement.appendChild(tr)
    }
})