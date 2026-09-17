
function buscarEGerarJSON() {

  /* this function is used with Google Sheet an App Script Extensions
   to extract s1 and s2 positions from each channel drawn in 
   Google Sheet
  */
  
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  var resultadoJSON = {}; // Cria o objeto vazio
  var resultadoJSON_indices = {}; // Cria o objeto vazio

  var canais = "ABCD"

  for (let letra of canais){

    for (var i = 1; i <= 32; i++) {
      var termoBusca = letra + i;
      var finder = sheet.createTextFinder(termoBusca).matchEntireCell(true).findNext();
      
      if (finder) {
        var numLinha = finder.getRow();
        var numColuna = finder.getColumn();
        var letraColuna = colunaParaLetra(numColuna); // Converte o número em letra
        
        // Adiciona ao JSON no formato desejado: "LinhaLetra" (ex: "1J")
        resultadoJSON[termoBusca] = { linha: numLinha, coluna: letraColuna };


        //Adiciona ao JSON o formato: IndiceLinhaIndiceColuna (ex: "19")
        resultadoJSON_indices[termoBusca]  = { s1: numColuna-1, s2: numLinha-1};

      } else {
        resultadoJSON[termoBusca] = "Não encontrado";
        resultadoJSON_indices[termoBusca] = "Não encontrado";
      }
    }

  }
  
  // Exibe o JSON final formatado no log
  Logger.log(JSON.stringify(resultadoJSON, null, 2));
  Logger.log(JSON.stringify(resultadoJSON_indices, null, 2));

  return resultadoJSON, resultadoJSON_indices;
}

// Função auxiliar para converter o número da coluna em letra (ex: 10 -> J)
function colunaParaLetra(coluna) {
  var letra = "";
  while (coluna > 0) {
    var resto = (coluna - 1) % 26;
    letra = String.fromCharCode(65 + resto) + letra;
    coluna = Math.floor((coluna - resto) / 26);
  }
  return letra;
}