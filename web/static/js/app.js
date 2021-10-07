const Measurement = {
  data() {
    return {
      counter: 0
    }
  },
  methods: {
      generate() {
          axios.get('/get_new_data')
              .then(function (response) {
                  const option = {
                      series: [{
                          type: 'line',
                          data: response.data
                      }]
                  }

                  lifetimeTrace.setOption(option)
              })
              .catch(function (error) {
                  console.log(error);
              });

          console.log(1)
      }
  }
}

const app = Vue.createApp(Measurement).mount('#measurement')
