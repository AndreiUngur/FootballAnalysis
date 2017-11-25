const uri = "http://localhost:5000/";

$("#favorites_form").submit(function(e) {
    e.preventDefault();
});

function compare_teams(form){
    var team_a = form.team_a.value;
    var team_b = form.team_b.value;

    $.get(uri+"team_a="+team_a+"&team_b="+team_b, function(data){
        console.log(data);
        var pass = data.pass_coords;
        var points = data.points_coords;
        var rush = data.rush_coords;
        var yards = data.yards_coords;

        analyze_numbers(team_a, team_b, "points",points);
        analyze_numbers(team_a, team_b, "yards_passing",pass);
        analyze_numbers(team_a, team_b, "rushing_yards",rush);
        analyze_numbers(team_a, team_b, "yards_overall",yards);
        
    });
}

function analyze_numbers(team_a, team_b, stat_name, data){
    var output_string = "";
    if(data[0] < 0){
        output_string += team_a+" allows less "+stat_name+" than "+team_b+". ";
    } else if(data[0] > 0) {
        output_string += team_a+" allows more "+stat_name+" than "+team_b+". ";    
    } else{
       output_string += team_a+" allows as many "+stat_name+" as "+team_b+". ";
    }

    if(data[1] < 0){
        output_string += team_a+" gains less "+stat_name+" than "+team_b+". ";
    } else if(data[1] > 0) {
        output_string += team_a+" gains more "+stat_name+" than "+team_b+". ";    
    } else{
       output_string += team_a+" gains as many "+stat_name+" as "+team_b+". ";
    }
    var obj = $("#"+stat_name).text(output_string+"\nraw data: "+data);
    obj.html(obj.html().replace(/\n/g,'<br/>'));
}