const lifetimeTrace = echarts.init(document.getElementById('lifetime-trace'))
const intensityTrace = echarts.init(document.getElementById('intensity-trace'))

const option = {
    xAxis: {
        data: [1, 2, 3, 4, 5, 6]
    },
    yAxis: {},
    series: [{
        type: 'line',
        data: [5, 20, 36, 10, 10, 20]
    }]
};

lifetimeTrace.setOption({
    title:{
        text:'Lifetime trace'
    }
})
lifetimeTrace.setOption(option)

intensityTrace.setOption({
    title:{
        text:'Intensity trace'
    }
})
intensityTrace.setOption(option)

