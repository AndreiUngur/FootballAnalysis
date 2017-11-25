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
        $("#team_name_A").text(team_a);
        $("#team_name_B").text(team_b);
        analyze_numbers(team_a, team_b, "points",points);
        analyze_numbers(team_a, team_b, "yards_passing",pass);
        analyze_numbers(team_a, team_b, "rushing_yards",rush);
        analyze_numbers(team_a, team_b, "yards_overall",yards);
        
    });
}
const less_good_def = "Less good defensively";
const less_good_of = "Less good offensively";
const good_def = "Better defensively";
const good_of = "Better offensively";
const equal = "The two teams are equal";
function analyze_numbers(team_a, team_b, stat_name, data){
    var output_string_A = "";
    var output_string_B = "";
    if(data[0] < 0){
        output_string_B += less_good_def+". ";
        output_string_A += good_def+". ";
    } else if(data[0] > 0) {
        output_string_A += less_good_def+". ";
        output_string_B += good_def+". ";  
    } else{
        output_string_A += equal+". ";
        output_string_B += equal+". ";
    }

    if(data[1] < 0){
        output_string_B += good_of+". ";
        output_string_A += less_good_of+". ";
    } else if(data[1] > 0) {
        output_string_A += good_of+". ";
        output_string_B += less_good_of+". ";
    } else{
        output_string_B += equal+". ";
        output_string_A += equal+". ";
     }
    $("#"+stat_name+"_A").text(output_string_A);
    $("#"+stat_name+"_B").text(output_string_B);
    $("#"+stat_name).text("Raw Data: "+data);
}