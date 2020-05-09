$(function(){
	$('#btn').click(function(){

		$.ajax({
			url: '/',
			data: $('form').serialize(),
			dataType:'json',
			type: 'POST',
			success: function(response){
				console.log(response);
				$("#category").text(response['category']);
			},
			error: function(error){
				console.log(error);
			}
		});
	});
});
