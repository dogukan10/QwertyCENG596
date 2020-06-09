$(function(){
	$('#btn').click(function(){

		$.ajax({
			url: '/',
			data: $('form').serialize(),
			dataType:'json',
			type: 'POST',
			success: function(response){
				console.log(response);
				$("#category-naive-bayes").text(response['category-naive-bayes']);
				$("#category-cnn").text(response['category-cnn']);
				$("#category-knn").text(response['category-knn']);
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});
